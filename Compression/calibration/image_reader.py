"""ImageCalibrationReader — feeds images to ``quantize_static``.

Matches the preprocessing used by ``src/inference/onnx_inferencer.py``
exactly so the statistics the quantiser observes are the same ones
the production inferencer will produce. Mismatched preprocessing
during calibration is the single biggest cause of INT8 accuracy
collapse, so we keep this in sync with the inferencer.

Two recipes, chosen automatically by caller:

    YOLO       : BGR → RGB → letterbox(model_h, model_w) → /255 → CHW
    RF-DETR    : BGR → RGB → stretch (model_h, model_w) → /255 →
                 (x - ImageNet_mean) / ImageNet_std → CHW
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _letterbox(frame: np.ndarray, target_h: int, target_w: int,
               pad: int = 114) -> np.ndarray:
    """Aspect-preserving pad to (target_h, target_w) with grey fill.

    Matches ``OptimizedONNXInferencer._letterbox`` in
    ``src/inference/onnx_inferencer.py`` — identical pad values, same
    INTER_LINEAR resize — so calibration samples arrive at the graph
    in exactly the shape production inference will produce.
    """
    h, w = frame.shape[:2]
    r = min(target_h / h, target_w / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(pad, pad, pad),
    )


class ImageCalibrationReader(CalibrationDataReader):
    """Iterate over images in a folder, yield tensors shaped for the model."""

    def __init__(
        self,
        image_dir: Path,
        input_name: str,
        input_shape: tuple[int, int, int, int],   # (N, C, H, W)
        is_rfdetr: bool,
        limit: int | None = 100,
    ):
        files = sorted(
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No images found under {image_dir}")
        if limit:
            files = files[:limit]

        _, _, h, w = input_shape
        self._files = files
        self._input_name = input_name
        self._h, self._w = int(h), int(w)
        self._is_rfdetr = is_rfdetr
        self._idx = 0

    def __len__(self) -> int:
        return len(self._files)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._is_rfdetr:
            resized = cv2.resize(rgb, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
            img = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        else:
            padded = _letterbox(rgb, self._h, self._w)
            img = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(img[np.newaxis, ...])

    def get_next(self) -> dict | None:
        """Return the next calibration sample or None when exhausted.

        Unreadable files are skipped silently rather than aborting the
        whole calibration run. The quantiser tolerates missing samples.
        """
        while self._idx < len(self._files):
            path = self._files[self._idx]
            self._idx += 1
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            return {self._input_name: self._preprocess(frame)}
        return None

    def rewind(self) -> None:
        self._idx = 0
