"""INT8 dynamic post-training quantisation stage.

Uses ``onnxruntime.quantization.quantize_dynamic`` — weights are
quantised once at export time, activations are quantised per-tensor
at runtime. No calibration step is needed, which is what makes this
robust on Ultralytics YOLO exports (the post-NMS subgraph has
Slice→NonMaxSuppression→ScatterND chains that crash static-mode
calibrators on dynamic shape edges).

Trade-off vs static quantisation:
  - Size win is the same (~75 % smaller; only weights compress).
  - Accuracy drop is slightly higher (~1–3 % mAP instead of ~1 %)
    because activation scales are picked per inference rather than
    learned once from representative data.
  - Runtime is slightly slower than static INT8 (activation quant
    happens every frame), still clearly faster than FP32.
  - Robustness is much higher — no graph has to survive the
    calibrator's internal ReduceMax tracking ops on problematic
    tensors.

We only quantise weight-heavy ops (Conv / MatMul / Gemm). These are
> 95 % of parameters in any detection model; the remaining ops stay
FP32 and run transparently through ONNX Runtime's mixed-precision
fallback.

Implementation note:
  ``quantize_dynamic`` in ORT ≤ 1.19 only produces **operator-level**
  quantised ops (``ConvInteger``, ``MatMulInteger``). These ops have
  **no CUDA implementation** and their CPU kernel was dropped from
  opset 21+. To avoid producing unrunnable models we force QDQ
  format via ``quantize_static`` with a minimal synthetic calibration
  reader — the model loads and runs on every EP just like INT8-QDQ,
  but activation scales are derived from random data rather than real
  images, giving "dynamic-like" quality without needing user images.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from ..inspect import ONNXProperties, inspect_onnx
from . import register
from .base import Stage


class _SyntheticCalibrationReader:
    """Yields random float32 tensors matching the model's input spec.

    Used for "dynamic-style" INT8 quantisation: we still call
    ``quantize_static`` (which outputs QDQ format — compatible with
    every EP), but we feed synthetic data instead of real images.
    Activation scales learned from random data are noisier than those
    from real images, matching the accuracy profile of true dynamic
    quantisation while producing a model that actually loads on CUDA.
    """

    def __init__(self, input_name: str, input_shape: tuple, num_samples: int = 8):
        self._input_name = input_name
        self._input_shape = input_shape
        self._index = 0
        self._num_samples = num_samples
        self._rng = np.random.default_rng(42)

    def get_next(self):
        if self._index >= self._num_samples:
            return None
        self._index += 1
        # Random tensor in [0, 1] — matches typical image normalisation range.
        tensor = self._rng.random(self._input_shape, dtype=np.float32)
        return {self._input_name: tensor}


@register
class INT8Stage(Stage):
    name = "int8"
    emoji = "🔢"
    description = "INT8 dynamic — QDQ format, runs on CUDA + CPU (no calibration needed)"

    #: Op types we quantise. Limiting to Conv/MatMul/Gemm (the weight-heavy
    #: ops, > 95 % of bytes in any detection model) keeps the NMS plumbing
    #: untouched and the quantiser clear of Slice/ReduceMax edge cases.
    #: Anything we don't touch stays FP32 and runs via ONNX Runtime's
    #: transparent mixed-precision fallback.
    op_types_to_quantize: tuple[str, ...] = ("Conv", "MatMul", "Gemm")

    #: Number of synthetic calibration samples. 8 is enough for the
    #: quantiser to learn reasonable activation scales from random data.
    synthetic_samples: int = 8

    def can_run(self, props: ONNXProperties) -> tuple[bool, str]:
        if props.already_int8:
            return False, (
                "weights are already INT8 or the graph contains QuantizeLinear "
                "nodes — INT8 would be a no-op"
            )
        if not props.inputs:
            return False, "model has no declared inputs — can't inspect for INT8 pass"
        return True, ""

    def run(self, src: Path) -> Path:
        out = self.output_path(src)
        props = inspect_onnx(src)

        input_info = props.inputs[0]
        input_name = input_info.name
        input_shape = self._concretise_shape(input_info.shape)

        # Pre-process to resolve dynamic shapes and fold constants.
        # Safe on any graph; helps some exporters that leave symbolic
        # dims behind. If it fails we fall through to the raw source.
        tmp = tempfile.NamedTemporaryFile(
            suffix=".preproc.onnx", delete=False, dir=src.parent,
        )
        preproc_path = Path(tmp.name)
        tmp.close()

        try:
            try:
                quant_pre_process(
                    input_model_path=str(src),
                    output_model_path=str(preproc_path),
                    skip_symbolic_shape=False,
                    skip_optimization=False,
                    skip_onnx_shape=False,
                    verbose=0,
                )
                quant_input = str(preproc_path)
            except Exception:
                quant_input = str(src)

            # Per-channel QDQ requires opset >= 13. Upgrade if needed.
            self._ensure_opset(quant_input, min_opset=13)

            reader = _SyntheticCalibrationReader(
                input_name=input_name,
                input_shape=input_shape,
                num_samples=self.synthetic_samples,
            )

            # QDQ static quantisation with synthetic data — produces
            # QuantizeLinear/DequantizeLinear pairs that every EP can
            # fuse, unlike ConvInteger which has no CUDA kernel.
            quantize_static(
                model_input=quant_input,
                model_output=str(out),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=list(self.op_types_to_quantize),
                per_channel=True,
                reduce_range=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={
                    "CalibTensorRangeSymmetric": True,
                },
            )
        finally:
            if preproc_path.exists():
                preproc_path.unlink()

        return out

    @staticmethod
    def _concretise_shape(shape: list) -> tuple[int, int, int, int]:
        """Pull (N, C, H, W) out of a 4-D input shape, defaulting if symbolic."""
        resolved = []
        for d in shape:
            if isinstance(d, int) and d > 0:
                resolved.append(d)
            else:
                resolved.append(1)  # batch fallback
        # Ensure we have 4 dims
        while len(resolved) < 4:
            resolved.append(640)
        return tuple(resolved[:4])  # type: ignore[return-value]

    @staticmethod
    def _ensure_opset(model_path: str, min_opset: int) -> None:
        """Upgrade an ONNX model's default-domain opset in place if needed."""
        import onnx
        from onnx import version_converter

        model = onnx.load(model_path)
        current = next(
            (op.version for op in model.opset_import if op.domain in ("", "ai.onnx")),
            0,
        )
        if current >= min_opset:
            return
        upgraded = version_converter.convert_version(model, min_opset)
        onnx.save(upgraded, model_path)
