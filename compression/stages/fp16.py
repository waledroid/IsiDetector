"""FP16 conversion stage.

Converts every float32 weight tensor in the ONNX graph to float16.
Halves the file size. Accuracy loss is typically < 0.5 % mAP on
detection tasks.

We pass ``keep_io_types=True`` so the network's inputs/outputs stay
float32 from the caller's perspective — the existing inference code
in ``src/inference/onnx_inferencer.py`` keeps working unchanged,
because the FP16 arithmetic is an internal detail of the graph.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

import onnx
from onnxconverter_common.float16 import convert_float_to_float16

from ..inspect import ONNXProperties
from ..ui import console
from . import register
from .base import Stage


@contextmanager
def _heartbeat(label: str, interval_s: float = 10.0):
    """Print a 'still alive' line every `interval_s` seconds inside a slow call.

    Some graphs push ``convert_float_to_float16`` into the 30–120s range;
    without feedback the operator can't tell a slow convert apart from a
    hang. A daemon thread emits one line per interval with the elapsed
    time. The thread exits cleanly on __exit__.
    """
    t0 = time.perf_counter()
    stop = threading.Event()

    def _tick() -> None:
        while not stop.wait(interval_s):
            console.print(
                f"   [dim]… still {label} "
                f"({time.perf_counter() - t0:.0f}s elapsed)…[/dim]"
            )

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


@register
class FP16Stage(Stage):
    name = "fp16"
    emoji = "🪶"
    description = "Convert FP32 weights to FP16 — halves size, minimal accuracy loss"

    def can_run(self, props: ONNXProperties) -> tuple[bool, str]:
        if props.already_fp16:
            return False, "weights are already float16 — running FP16 would be a no-op"
        if props.weight_dtype == "int8":
            return False, (
                "weights are INT8; downcasting from INT8 to FP16 isn't supported. "
                "Run FP16 *before* INT8 if you want the combination."
            )
        return True, ""

    #: Ops kept at FP32 even when everything else converts to FP16.
    #:
    #: Keep this list **minimal**. Every entry forces the converter to
    #: insert Cast pairs around *every* occurrence of that op type in the
    #: graph and re-validate the graph per insertion. Blocking ubiquitous
    #: ops like ``Mul``/``Add``/``Cast`` (one per fused-BatchNorm layer in
    #: a YOLO) pushes the conversion into O(n²) territory: a 10 MB model
    #: stalls the converter for minutes.
    #:
    #: The four ops below cover the real precision/compatibility hazards:
    #:
    #: * ``Resize`` / ``Upsample`` — Ultralytics YOLO exports wire Cast
    #:   nodes around these in a way that produces a load-time
    #:   "Type (tensor(float)) ... expected (tensor(float16))" error if
    #:   they're converted to FP16.
    #: * ``ReduceSum`` / ``TopK`` — FP16 precision is insufficient for
    #:   the accumulation / sorting they do (onnxconverter-common docs
    #:   flag both).
    #:
    #: The post-NMS region (``NonMaxSuppression`` + surrounding
    #: arithmetic) is **not** blocked here. With ``keep_io_types=True``
    #: the model's outputs stay FP32, so the NMS coordinate Mul/Add
    #: nodes at the boundary receive FP32 inputs from the outside and
    #: don't need explicit blocking.
    fp16_op_block_list: tuple[str, ...] = (
        "Resize", "Upsample",
        "ReduceSum", "TopK",
    )

    def run(self, src: Path) -> Path:
        out = self.output_path(src)

        # Per-phase prints so the operator can see which step is taking time.
        # On large YOLO exports the load+convert can each take 10–30s with no
        # feedback otherwise, which looks identical to a hang.
        size_mb = src.stat().st_size / (1024 * 1024)
        console.print(f"   [dim]⏳ loading {src.name} ({size_mb:.0f} MB)…[/dim]")
        t0 = time.perf_counter()
        model = onnx.load(str(src))
        console.print(f"   [dim]✓ loaded in {time.perf_counter() - t0:.1f}s[/dim]")

        # keep_io_types=True  → network inputs/outputs stay float32.
        #                       Callers never need to know about the conversion.
        # op_block_list       → ops that misbehave at FP16 stay at FP32; the
        #                       converter inserts Cast pairs around them.
        # disable_shape_infer → TRUE. The converter's internal ONNX shape
        #                       inference pass is expensive and on some YOLO
        #                       graphs gets stuck for tens of seconds even on
        #                       a 10 MB model. We strip value_info and re-run
        #                       shape inference ourselves in phase 3 anyway,
        #                       so doing it twice is pure duplicate work.
        console.print(
            f"   [dim]⏳ converting to FP16 "
            f"(blocklist: {len(self.fp16_op_block_list)} op types)…[/dim]"
        )
        t0 = time.perf_counter()
        with _heartbeat("converting to FP16"):
            converted = convert_float_to_float16(
                model,
                keep_io_types=True,
                disable_shape_infer=True,
                op_block_list=list(self.fp16_op_block_list),
            )
        console.print(f"   [dim]✓ converted in {time.perf_counter() - t0:.1f}s[/dim]")

        # `convert_float_to_float16` rewrites weight initializers and
        # inserts boundary Cast pairs around blocklisted ops, but it
        # does NOT touch the ``to`` attribute of pre-existing Cast nodes
        # baked into the original export. On Ultralytics YOLO graphs
        # there is at least one such Cast (``/model.23/Cast_2``) whose
        # output feeds a Concat alongside three FP16 tensors. After
        # conversion that Cast still emits FP32, the Concat sees mixed
        # dtypes, and ORT refuses to load with::
        #
        #   Type Error: Type parameter (T) of Optype (Concat) bound to
        #   different types (tensor(float16) and tensor(float)) in node
        #   (/model.23/Concat_7).
        #
        # Sweep the graph and rewrite orphaned ``Cast(to=FP32)`` nodes
        # to ``Cast(to=FP16)`` unless they're legitimate boundary casts
        # (feeding a blocklisted op, or a model output under keep_io_types).
        n_fixed = self._fix_orphan_fp32_casts(converted)
        if n_fixed:
            console.print(
                f"   [dim]✓ rewrote {n_fixed} orphan FP32→FP16 cast(s)[/dim]"
            )

        # The converter appends new FP16 value_info entries without
        # removing the matching FP32 ones, so the graph ends up with
        # duplicate entries for the same tensor name (one FLOAT, one
        # FLOAT16). ORT's loader rejects the model with:
        #   Type (tensor(float)) of output arg (.../Resize_output_0) of
        #   node (.../Resize_output_cast0) does not match expected type
        #   (tensor(float16)).
        # Strip the graph's value_info, then re-run shape inference so
        # types get re-derived once from the (now correct) node graph.
        console.print("   [dim]⏳ re-inferring shapes…[/dim]")
        t0 = time.perf_counter()
        del converted.graph.value_info[:]
        try:
            with _heartbeat("re-inferring shapes"):
                converted = onnx.shape_inference.infer_shapes(converted)
        except Exception:
            # Shape inference can fail on graphs with custom ops; the
            # original model may still load without value_info, so don't
            # let this step abort the conversion.
            pass
        console.print(f"   [dim]✓ shapes inferred in {time.perf_counter() - t0:.1f}s[/dim]")

        console.print(f"   [dim]⏳ saving {out.name}…[/dim]")
        t0 = time.perf_counter()
        onnx.save(converted, str(out))
        console.print(f"   [dim]✓ saved in {time.perf_counter() - t0:.1f}s[/dim]")
        return out

    def _fix_orphan_fp32_casts(self, model) -> int:
        """Rewrite leftover ``Cast(to=FP32)`` nodes the converter missed.

        ``convert_float_to_float16`` does not touch the ``to`` attribute
        of pre-existing Cast nodes from the original export. When such
        a Cast feeds a now-FP16 op, its FP32 output causes a load-time
        Type Error in ORT. Walk the graph and rewrite each orphan to
        ``to=FP16``, preserving the cases where FP32 is correct:

        * the consumer is a blocklisted op (Resize/Upsample/TopK/
          ReduceSum) that legitimately stayed FP32 — this is the
          converter's own boundary cast,
        * the cast's output is a model output (``keep_io_types=True``
          forces FP32 at the IO boundary).

        Returns the number of casts rewritten.
        """
        FP32, FP16 = 1, 10
        consumers: dict[str, list] = {}
        for node in model.graph.node:
            for inp in node.input:
                consumers.setdefault(inp, []).append(node)
        output_names = {o.name for o in model.graph.output}
        blocklisted = set(self.fp16_op_block_list)

        fixed = 0
        for node in model.graph.node:
            if node.op_type != "Cast":
                continue
            to_attr = next(
                (a for a in node.attribute if a.name == "to"), None
            )
            if to_attr is None or to_attr.i != FP32:
                continue
            out_name = node.output[0]
            if out_name in output_names:
                continue
            if any(c.op_type in blocklisted
                   for c in consumers.get(out_name, [])):
                continue
            to_attr.i = FP16
            fixed += 1
        return fixed
