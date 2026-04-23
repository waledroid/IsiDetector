"""Benchmark runner — measure size + FPS + p95 latency per ONNX variant.

Runs a fixed preprocessed tensor through each ``.onnx`` file in a model
group, timed with ``time.perf_counter()``. The first few iterations are
discarded (warmup) so we don't count CUDA kernel-compilation time.

Single-frame repeat measurement because we want pure kernel-speed
numbers — operators care about "once the belt is running, how fast is
it?" not "what's the cold-start cost?". For cold-start we'd measure
session construction time instead, which is a separate concern.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import onnxruntime as ort

from .calibration.image_reader import _IMAGENET_MEAN, _IMAGENET_STD, _letterbox
from .discovery import ModelGroup, ONNXFile

#: Output names that mark the model as RF-DETR family (drives preprocessing).
_RFDETR_OUT = {"dets", "pred_logits", "pred_boxes", "bboxes", "labels"}

#: Directories we'll search for a sample frame, in priority order.
_SAMPLE_FRAME_DIRS = (
    Path("isidet/data/universal_dataset/images/val"),
    Path("isidet/data/universal_dataset/images/train"),
    Path("isidet/data/isi_3k_dataset/images/val"),
    Path("isidet/data/isi_3k_dataset/images/train"),
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BenchmarkResult:
    """One variant's benchmark outcome."""

    file: ONNXFile
    size_mb: float
    iterations: int
    median_ms: float
    p95_ms: float
    fps: float
    provider: str               # EP actually used ('CUDAExecutionProvider' or 'CPUExecutionProvider')
    error: str | None = None    # populated if the variant failed to load/run


# ── Helpers ─────────────────────────────────────────────────────────────────

def _is_rfdetr(session: ort.InferenceSession) -> bool:
    return any(o.name in _RFDETR_OUT for o in session.get_outputs())


def _preprocess(frame: np.ndarray, h: int, w: int, is_rfdetr: bool) -> np.ndarray:
    """Mirror ``src/inference/onnx_inferencer.py``'s preprocess.

    YOLO:  letterbox + /255
    RF-DETR: stretch + /255 + ImageNet norm
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if is_rfdetr:
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        x = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    else:
        padded = _letterbox(rgb, h, w)
        x = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[np.newaxis, ...])


def _pick_sample_frame() -> tuple[np.ndarray, Path | None]:
    """Find a real image to benchmark on; fall back to a grey frame.

    Returns ``(frame, path_or_None)`` — path is ``None`` when we had
    to synthesise a grey frame so the caller can mention it in logs.
    """
    for d in _SAMPLE_FRAME_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                frame = cv2.imread(str(f))
                if frame is not None:
                    return frame, f
    return np.full((720, 1280, 3), 128, dtype=np.uint8), None


def _concretise(shape: list[int | str]) -> tuple[int, int]:
    """Pull (H, W) out of a 4-D NCHW input shape, defaulting if symbolic."""
    h = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 640
    w = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 640
    return int(h), int(w)


# ── Public API ──────────────────────────────────────────────────────────────

def _build_session(
    path: str, opts: ort.SessionOptions, allow_cuda: bool
) -> ort.InferenceSession:
    """Build a session, falling back to CPU-only on CUDA load failure.

    Dynamic-INT8 models contain ConvInteger/MatMulInteger ops that only
    have a CPU implementation. Loading them on CUDA throws "not
    implemented" at session construction. We catch that and retry with a
    CPU-only provider list — the caller's BenchmarkResult tells the user
    which EP actually ran the variant.
    """
    available = ort.get_available_providers()
    if allow_cuda and "CUDAExecutionProvider" in available:
        try:
            return ort.InferenceSession(
                path, sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            # Drop through to CPU-only retry below.
            pass
    return ort.InferenceSession(
        path, sess_options=opts, providers=["CPUExecutionProvider"],
    )


def benchmark_file(
    onnx_file: ONNXFile,
    frame: np.ndarray,
    warmup: int = 5,
    iterations: int = 50,
) -> BenchmarkResult:
    """Benchmark a single ONNX file on ``frame``. Never raises."""
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        session = _build_session(str(onnx_file.path), opts, allow_cuda=True)

        inp = session.get_inputs()[0]
        if len(inp.shape) != 4:
            raise ValueError(f"input shape {inp.shape} is not NCHW")
        h, w = _concretise(inp.shape)
        is_rfdetr = _is_rfdetr(session)
        tensor = _preprocess(frame, h, w, is_rfdetr)

        # Warmup — discarded. Covers CUDA kernel JIT + cuDNN algo selection.
        for _ in range(max(1, warmup)):
            session.run(None, {inp.name: tensor})

        # Measured run.
        timings_ms: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            session.run(None, {inp.name: tensor})
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        timings_ms.sort()
        med = median(timings_ms)
        p95 = timings_ms[min(len(timings_ms) - 1, int(len(timings_ms) * 0.95))]
        fps = 1000.0 / med if med > 0 else 0.0

        return BenchmarkResult(
            file=onnx_file,
            size_mb=onnx_file.size_mb,
            iterations=iterations,
            median_ms=med,
            p95_ms=p95,
            fps=fps,
            provider=session.get_providers()[0],
        )
    except Exception as e:  # noqa: BLE001 — failures become data, never crashes
        return BenchmarkResult(
            file=onnx_file,
            size_mb=onnx_file.size_mb,
            iterations=0,
            median_ms=0.0,
            p95_ms=0.0,
            fps=0.0,
            provider="-",
            error=str(e),
        )


def run_group_benchmark(
    group: ModelGroup,
    iterations: int = 50,
    warmup: int = 5,
    on_progress=None,
) -> tuple[list[BenchmarkResult], Path | None]:
    """Benchmark every variant in ``group``. Returns (results, sample_path).

    ``on_progress`` is an optional callable ``(done, total, filename)``
    called after each variant completes — the CLI uses it to drive a
    rich Progress bar.
    """
    frame, sample_path = _pick_sample_frame()
    results: list[BenchmarkResult] = []
    total = len(group.files)
    for i, onnx_file in enumerate(group.files, start=1):
        r = benchmark_file(onnx_file, frame, warmup=warmup, iterations=iterations)
        results.append(r)
        if on_progress is not None:
            on_progress(i, total, onnx_file.path.name)
    return results, sample_path
