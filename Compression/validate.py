"""Validation runner — diff detections between a baseline and compressed variant.

The purpose is a safety net for compressions that could silently lose
accuracy (INT8 above all). Rather than measure mAP with ground-truth
labels (which we don't always have on the target site), we compare
two inferencers on the same frames and report how much the compressed
variant's detections drift from the baseline.

We reuse ``src/inference/onnx_inferencer.OptimizedONNXInferencer`` so
every preprocessing and postprocessing decision matches the live web
app exactly. Matched preprocessing is the difference between a metric
that predicts production behaviour and one that lies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.inference.onnx_inferencer import OptimizedONNXInferencer

from .discovery import ONNXFile

#: Where to pull sample frames from. Same priority list as the benchmark
#: runner so both tools operate on the same input distribution.
_SAMPLE_DIRS = (
    Path("data/universal_dataset/images/val"),
    Path("data/universal_dataset/images/train"),
    Path("data/isi_3k_dataset/images/val"),
    Path("data/isi_3k_dataset/images/train"),
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ValidationResult:
    """Aggregated accuracy-drift metrics between baseline and candidate."""

    baseline_file: ONNXFile
    candidate_file: ONNXFile

    # Sample set
    n_frames_checked: int
    sample_dir: Path | None

    # Raw detection counts
    baseline_detections: int
    candidate_detections: int

    # Matched pairs (greedy IoU ≥ iou_threshold)
    matched_pairs: int
    iou_threshold: float

    # Per-match drift metrics
    class_match_rate: float       # (same class) / matched
    mean_box_drift_px: float      # bbox-centre L2 distance
    max_box_drift_px: float
    mean_conf_delta: float        # base.conf − cand.conf, positive = lost confidence

    # Verdict
    verdict: str                  # 'good' | 'acceptable' | 'degraded' | 'broken' | 'no-baseline'
    verdict_reason: str

    @property
    def keep_rate(self) -> float:
        if self.baseline_detections == 0:
            return 0.0
        return self.candidate_detections / self.baseline_detections

    @property
    def match_rate(self) -> float:
        if self.baseline_detections == 0:
            return 0.0
        return self.matched_pairs / self.baseline_detections


# ── Geometry helpers ────────────────────────────────────────────────────────

def _iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU on [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _greedy_match(
    a_boxes: np.ndarray,
    b_boxes: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Greedy 1-to-1 matching: each baseline claims its best unused candidate."""
    matches: list[tuple[int, int]] = []
    used_b: set[int] = set()
    for i, a in enumerate(a_boxes):
        best_iou, best_j = 0.0, -1
        for j, b in enumerate(b_boxes):
            if j in used_b:
                continue
            iou = _iou_xyxy(a, b)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            matches.append((i, best_j))
            used_b.add(best_j)
    return matches


def _box_centre_drift_px(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ca_x = (box_a[0] + box_a[2]) / 2.0
    ca_y = (box_a[1] + box_a[3]) / 2.0
    cb_x = (box_b[0] + box_b[2]) / 2.0
    cb_y = (box_b[1] + box_b[3]) / 2.0
    return float(np.hypot(ca_x - cb_x, ca_y - cb_y))


# ── Frame sourcing ──────────────────────────────────────────────────────────

def _find_sample_dir() -> Path | None:
    for d in _SAMPLE_DIRS:
        if d.is_dir() and any(
            f.is_file() and f.suffix.lower() in _IMAGE_EXTS for f in d.iterdir()
        ):
            return d
    return None


def _collect_frames(sample_dir: Path, limit: int) -> list[np.ndarray]:
    paths = sorted(
        f for f in sample_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
    )
    frames: list[np.ndarray] = []
    for p in paths:
        if len(frames) >= limit:
            break
        frame = cv2.imread(str(p))
        if frame is not None:
            frames.append(frame)
    return frames


# ── Verdict logic ───────────────────────────────────────────────────────────

def _decide_verdict(result: dict) -> tuple[str, str]:
    """Classify the run from the raw metrics. Thresholds documented inline."""
    if result["baseline_detections"] == 0:
        return "no-baseline", "baseline produced no detections on sample set"

    keep = result["keep_rate"]
    match = result["match_rate"]
    cls = result["class_match_rate"]

    # ≥97 % detections kept AND ≥95 % IoU-matched AND ≥98 % class agreement
    if keep >= 0.97 and match >= 0.95 and cls >= 0.98:
        return "good", "negligible regression from baseline"

    # ≥90 % detections kept AND ≥85 % matched AND ≥95 % class agreement
    if keep >= 0.90 and match >= 0.85 and cls >= 0.95:
        return "acceptable", f"within tolerance ({keep:.0%} kept, {match:.0%} IoU-matched, {cls:.0%} class agreement)"

    if keep >= 0.70:
        return "degraded", f"measurable regression ({keep:.0%} kept)"

    return "broken", f"major regression ({keep:.0%} detections kept)"


# ── Public API ──────────────────────────────────────────────────────────────

def validate_pair(
    baseline_file: ONNXFile,
    candidate_file: ONNXFile,
    n_frames: int = 50,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.3,
    on_progress=None,
) -> ValidationResult:
    """Run baseline and candidate on the same frames, diff the detections."""
    sample_dir = _find_sample_dir()
    if sample_dir is None:
        raise FileNotFoundError(
            "No sample frames found under "
            + ", ".join(str(d) for d in _SAMPLE_DIRS)
        )

    frames = _collect_frames(sample_dir, n_frames)
    if not frames:
        raise FileNotFoundError(f"No readable images in {sample_dir}")

    base = OptimizedONNXInferencer(str(baseline_file.path), conf_threshold=conf_threshold)
    cand = OptimizedONNXInferencer(str(candidate_file.path), conf_threshold=conf_threshold)

    base_total = 0
    cand_total = 0
    matched = 0
    class_matches = 0
    box_drifts: list[float] = []
    conf_deltas: list[float] = []

    for idx, frame in enumerate(frames, start=1):
        try:
            base_det = base.predict_frame(frame.copy())
            cand_det = cand.predict_frame(frame.copy())
        except Exception:  # noqa: BLE001 — one bad frame shouldn't kill the run
            if on_progress is not None:
                on_progress(idx, len(frames))
            continue

        base_boxes = np.asarray(base_det.xyxy) if base_det.xyxy is not None else np.zeros((0, 4))
        cand_boxes = np.asarray(cand_det.xyxy) if cand_det.xyxy is not None else np.zeros((0, 4))
        base_total += len(base_boxes)
        cand_total += len(cand_boxes)

        if len(base_boxes) and len(cand_boxes):
            pairs = _greedy_match(base_boxes, cand_boxes, iou_threshold)
            matched += len(pairs)
            for a_i, b_j in pairs:
                # Class-id agreement (lenient if either side lacks class info)
                if (
                    base_det.class_id is not None
                    and cand_det.class_id is not None
                    and len(base_det.class_id) > a_i
                    and len(cand_det.class_id) > b_j
                ):
                    if int(base_det.class_id[a_i]) == int(cand_det.class_id[b_j]):
                        class_matches += 1
                # Bbox centre drift
                box_drifts.append(_box_centre_drift_px(base_boxes[a_i], cand_boxes[b_j]))
                # Confidence delta (positive = compressed model is less confident)
                if (
                    base_det.confidence is not None
                    and cand_det.confidence is not None
                    and len(base_det.confidence) > a_i
                    and len(cand_det.confidence) > b_j
                ):
                    conf_deltas.append(
                        float(base_det.confidence[a_i]) - float(cand_det.confidence[b_j])
                    )

        if on_progress is not None:
            on_progress(idx, len(frames))

    # Aggregate
    class_match_rate = (class_matches / matched) if matched else 0.0
    mean_drift = float(np.mean(box_drifts)) if box_drifts else 0.0
    max_drift = float(np.max(box_drifts)) if box_drifts else 0.0
    mean_conf = float(np.mean(conf_deltas)) if conf_deltas else 0.0

    keep_rate = (cand_total / base_total) if base_total else 0.0
    match_rate = (matched / base_total) if base_total else 0.0

    verdict, reason = _decide_verdict({
        "baseline_detections": base_total,
        "keep_rate": keep_rate,
        "match_rate": match_rate,
        "class_match_rate": class_match_rate,
    })

    return ValidationResult(
        baseline_file=baseline_file,
        candidate_file=candidate_file,
        n_frames_checked=len(frames),
        sample_dir=sample_dir,
        baseline_detections=base_total,
        candidate_detections=cand_total,
        matched_pairs=matched,
        iou_threshold=iou_threshold,
        class_match_rate=class_match_rate,
        mean_box_drift_px=mean_drift,
        max_box_drift_px=max_drift,
        mean_conf_delta=mean_conf,
        verdict=verdict,
        verdict_reason=reason,
    )
