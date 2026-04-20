"""INT8 static post-training quantisation in QDQ format.

Same ~75 % size reduction as dynamic INT8, but the output runs on
**every ONNX Runtime execution provider** — CUDA, CPU, OpenVINO,
TensorRT — because QDQ wraps each quantised op in
QuantizeLinear/DequantizeLinear pairs around regular Conv/MatMul.
Every EP knows how to fuse those pairs natively.

Trade-off vs the dynamic stage:
  - Needs calibration images. Same source priority as we use for the
    benchmark sample frame: ``data/universal_dataset/images/val``
    first, then siblings.
  - Runs the FP32 model during export to learn activation scales.
    This is exactly the step that crashed earlier on Ultralytics YOLO,
    so we apply several robustness measures (see ``run`` for detail).

Robustness measures:
  - ``op_types_to_quantize`` restricted to Conv / MatMul / Gemm so the
    calibrator never inserts internal ``ReduceMax`` tracking ops on
    Slice outputs in the post-NMS region.
  - ``CalibrationMethod.Percentile`` is more tolerant of zero-element
    or outlier-heavy tensors than ``MinMax``.
  - ``CalibTensorRangeSymmetric=True`` matches INT8's signed range,
    avoiding asymmetric clipping.
  - ``quant_pre_process`` resolves dynamic shapes before calibration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
import numpy as np

from onnxruntime.quantization.calibrate import HistogramCollector, TensorsData
from onnxruntime.quantization.shape_inference import quant_pre_process


def _patch_tensors_data_setitem() -> None:
    """Make ``TensorsData.__setitem__`` tolerant of unseen keys.

    ORT 1.19's ``base_quantizer.adjust_tensor_ranges`` unconditionally
    pins every Softmax output's range to ``[0, 1]``, regardless of
    whether that tensor was part of the calibration set. When
    ``op_types_to_quantize`` excludes Softmax (as we do — only Conv,
    MatMul, Gemm quantise), the Softmax output never entered the
    calibrator's dict, and the default ``__setitem__`` raises
    ``RuntimeError: Only an existing tensor can be modified``.

    The upstream behaviour is overly defensive: a missing key is not
    a correctness issue for the adjust-phase, it just means there's
    nothing to adjust. We patch once at import time to convert the
    error into a silent insert, matching the fix that landed upstream
    after 1.19.
    """
    if getattr(TensorsData, "_isitec_patched", False):
        return

    def _tolerant_setitem(self, key, value):
        self.data[key] = value

    TensorsData.__setitem__ = _tolerant_setitem  # type: ignore[assignment]
    TensorsData._isitec_patched = True           # type: ignore[attr-defined]


_patch_tensors_data_setitem()


def _patch_histogram_collector() -> None:
    """Skip empty / non-finite tensors during histogram calibration.

    YOLO seg graphs produce intermediate tensors whose shape contains
    a ``0`` dimension on certain frames (e.g. zero-detection branches
    of conditional NMS sub-graphs). When ORT's ``HistogramCollector``
    calls ``np.histogram`` on such an array, numpy raises
    ``autodetected range of [] is not finite``. Same failure mode for
    tensors that contain ``inf``/``NaN``. These tensors carry no
    useful calibration signal, so we filter them out before they reach
    numpy — quantisation falls back to default ranges downstream.
    """
    if getattr(HistogramCollector, "_isitec_patched", False):
        return

    def _sanitise(name_to_arr):
        cleaned = {}
        for tensor, data in name_to_arr.items():
            if isinstance(data, list):
                kept = [a for a in data if a.size > 0 and np.isfinite(a).all()]
                if not kept:
                    continue
                cleaned[tensor] = kept
            else:
                if data.size == 0 or not np.isfinite(data).all():
                    continue
                cleaned[tensor] = data
        return cleaned

    orig_collect_value = HistogramCollector.collect_value
    orig_collect_absolute_value = HistogramCollector.collect_absolute_value

    def collect_value(self, name_to_arr):
        return orig_collect_value(self, _sanitise(name_to_arr))

    def collect_absolute_value(self, name_to_arr):
        return orig_collect_absolute_value(self, _sanitise(name_to_arr))

    HistogramCollector.collect_value = collect_value           # type: ignore[assignment]
    HistogramCollector.collect_absolute_value = collect_absolute_value  # type: ignore[assignment]
    HistogramCollector._isitec_patched = True                  # type: ignore[attr-defined]


_patch_histogram_collector()

from ..calibration.image_reader import ImageCalibrationReader
from ..inspect import ONNXProperties, inspect_onnx
from . import register
from .base import Stage

_CALIB_DIRS: tuple[Path, ...] = (
    Path("data/universal_dataset/images/val"),
    Path("data/universal_dataset/images/train"),
    Path("data/isi_3k_dataset/images/val"),
    Path("data/isi_3k_dataset/images/train"),
    Path("data/rfdetr_dataset/valid/images"),
    Path("data/rfdetr_dataset/train/images"),
)

_RFDETR_OUTPUT_NAMES = {"dets", "pred_logits", "pred_boxes", "bboxes", "labels"}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@register
class INT8QDQStage(Stage):
    name = "int8qdq"
    emoji = "🎯"
    description = "INT8 QDQ — runs on CUDA / CPU / OpenVINO / TRT (needs calibration)"

    #: Calibration sample count. 32 keeps activation-tensor RAM in check
    #: for RF-DETR-class transformers on WSL; raise toward 100 only on
    #: hosts with >32 GB free RAM.
    calibration_samples: int = 32

    #: Restrict quantisation to weight-heavy ops to avoid the calibrator
    #: profiling Slice / ScatterND / NonMaxSuppression chains.
    op_types_to_quantize: tuple[str, ...] = ("Conv", "MatMul", "Gemm")

    def can_run(self, props: ONNXProperties) -> tuple[bool, str]:
        if props.already_int8:
            return False, (
                "weights are already INT8 or graph already has QuantizeLinear "
                "nodes — INT8 QDQ would be a no-op"
            )
        if not props.inputs or len(props.inputs[0].shape) != 4:
            return False, "expected a 4-D NCHW image input for calibration"
        if self._find_calibration_dir() is None:
            cands = ", ".join(str(p) for p in _CALIB_DIRS)
            return False, (
                f"no calibration images. Drop ~100 representative belt images "
                f"into one of: {cands}"
            )
        return True, ""

    def run(self, src: Path) -> Path:
        out = self.output_path(src)
        props = inspect_onnx(src)
        calib_dir = self._find_calibration_dir()
        assert calib_dir is not None  # guarded by can_run

        input_name = props.inputs[0].name
        input_shape = self._concretise_shape(props.inputs[0].shape)
        is_rfdetr = self._is_rfdetr(props)

        # Pre-process: symbolic shape inference + ONNX optimiser. Resolves
        # dynamic-shape ambiguities the calibrator can't handle. Same step
        # as the dynamic INT8 stage uses.
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
                    skip_optimization=True,
                    skip_onnx_shape=False,
                    verbose=0,
                )
                quant_input = str(preproc_path)
            except Exception:
                quant_input = str(src)

            # Per-channel QDQ requires opset >= 13. Ultralytics exports
            # YOLO at opset 12 by default, so upgrade in place if needed.
            self._ensure_opset(quant_input, min_opset=13)

            reader = ImageCalibrationReader(
                image_dir=calib_dir,
                input_name=input_name,
                input_shape=input_shape,
                is_rfdetr=is_rfdetr,
                limit=self.calibration_samples,
            )

            # Skip nodes whose surrounding graph the QDQ optimiser mangles.
            # YOLO's DFL head (Softmax→Transpose→Conv with fixed 1..reg_max
            # weights) is fused internally during QDQ placement, leaving
            # the Softmax output name dangling. Its 16-channel Conv barely
            # benefits from INT8, so exclude it.
            nodes_to_exclude = self._find_nodes_to_exclude(quant_input)

            quantize_static(
                model_input=quant_input,
                model_output=str(out),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=list(self.op_types_to_quantize),
                nodes_to_exclude=nodes_to_exclude,
                per_channel=True,
                reduce_range=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.Percentile,
                extra_options={
                    "CalibTensorRangeSymmetric": True,
                },
            )
        finally:
            if preproc_path.exists():
                preproc_path.unlink()

        return out

    # ── helpers (kept private, mirrored from dynamic stage for parity) ───

    @staticmethod
    def _find_calibration_dir() -> Path | None:
        for p in _CALIB_DIRS:
            if p.is_dir() and any(
                f.suffix.lower() in _IMAGE_EXTS
                for f in p.iterdir() if f.is_file()
            ):
                return p
        return None

    @staticmethod
    def _concretise_shape(shape: list) -> tuple[int, int, int, int]:
        resolved = []
        for d in shape:
            if isinstance(d, int) and d > 0:
                resolved.append(d)
            else:
                resolved.append(1)  # batch fallback
        return tuple(resolved)  # type: ignore[return-value]

    @staticmethod
    def _is_rfdetr(props: ONNXProperties) -> bool:
        return any(o.name in _RFDETR_OUTPUT_NAMES for o in props.outputs)

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

    @staticmethod
    def _find_nodes_to_exclude(model_path: str) -> list[str]:
        """Return node names that break QDQ placement on YOLO graphs.

        The DFL head in Ultralytics YOLO (Softmax → Transpose → Conv with
        fixed 1..reg_max weights) trips a QDQ fusion bug: the Softmax
        output tensor name is rewritten mid-pass and the quantiser raises
        ``Only an existing tensor can be modified``. Skipping the DFL
        Conv costs essentially zero accuracy — its 16-channel weights are
        a fixed coefficient vector, not learned features.
        """
        import onnx

        model = onnx.load(model_path)
        return [n.name for n in model.graph.node if "/dfl/" in n.name]
