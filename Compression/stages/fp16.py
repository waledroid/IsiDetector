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

import time
from pathlib import Path

import onnx
from onnxconverter_common.float16 import convert_float_to_float16

from ..inspect import ONNXProperties
from ..ui import console
from . import register
from .base import Stage


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
    #: ``Resize`` and ``Upsample`` produce broken cast wiring around the
    #: output in some Ultralytics YOLO exports — leaving them at FP32 fixes
    #: the load-time "Type Error: Type (tensor(float)) ... expected
    #: (tensor(float16))" you'd otherwise hit.
    #: ``ReduceSum`` and ``TopK`` are blocked because FP16 precision is
    #: insufficient for the kind of accumulation/sorting they do — a known
    #: pitfall flagged in the onnxconverter-common docs.
    #: The post-NMS region (``NonMaxSuppression`` → ``NonZero`` → ``Cast``
    #: → ``Mul`` / ``Add`` / ``Gather`` / etc.) is blocked wholesale: the
    #: converter converts scalar constants to FP16 while Cast outputs
    #: remain FP32, producing mixed-type ``Mul`` nodes that ORT rejects
    #: with "Type parameter (T) bound to different types". These ops
    #: carry no weight tensors so there's zero size penalty.
    fp16_op_block_list: tuple[str, ...] = (
        # Interpolation ops — broken cast wiring on YOLO exports:
        "Resize", "Upsample",
        # Precision-sensitive accumulation / sorting:
        "ReduceSum", "TopK",
        # Post-NMS coordinate arithmetic & indexing — partial FP16
        # conversion creates float32/float16 mismatches at Mul nodes:
        "Mul", "Add", "Sub", "Div",
        "Cast", "NonZero", "NonMaxSuppression",
        "Gather", "GatherND", "ScatterND",
        "Greater", "Where", "ArgMax",
        "Shape", "Expand", "Pad",
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
        # disable_shape_infer → off, so FP16 propagation is shape-aware.
        console.print(
            f"   [dim]⏳ converting to FP16 "
            f"(blocklist: {len(self.fp16_op_block_list)} op types)…[/dim]"
        )
        t0 = time.perf_counter()
        converted = convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=False,
            op_block_list=list(self.fp16_op_block_list),
        )
        console.print(f"   [dim]✓ converted in {time.perf_counter() - t0:.1f}s[/dim]")

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
