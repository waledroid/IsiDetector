"""ONNX file discovery and grouping.

Scans the project root for ``*.onnx`` files and groups them by their
"run directory" — the directory that represents a single training
run or model export. This way ``best.onnx`` and ``best.sim.onnx``
produced from the same training appear as one entry with two files.

No external deps — stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Directory names we never want to walk into.
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "site",          # mkdocs build output
    "uploads",       # operator-uploaded videos
    ".deployment.env",
}

#: Directory names that indicate a "family" of model.
_FAMILY_HINTS = {
    "yolo": "yolo",
    "yolov26": "yolo",
    "rfdetr": "rfdetr",
}

#: Filename-token → variant-label map. Multiple matches concatenate with '+'
#: in the order they appear in the filename, so ``best.fp16.sim.onnx`` reads
#: as ``fp16+sim`` while ``best.sim.fp16.onnx`` reads as ``sim+fp16``.
_TOKEN_TAG = {
    "fp16":    "fp16",
    "int8":    "int8",         # dynamic-format INT8 (CPU-only)
    "int8qdq": "int8-qdq",     # static QDQ-format INT8 (cross-EP)
    "quant":   "int8",
    "sim":     "sim",
}


@dataclass
class ONNXFile:
    """One ONNX artefact on disk."""
    path: Path               # absolute path
    rel: Path                # relative to the project root
    size_mb: float           # file size in MB (2 decimals)
    variant: str             # 'base' | 'sim' | 'fp16' | 'int8' | 'rfdetr-export'


@dataclass
class ModelGroup:
    """A set of ONNX files produced by the same training run / export."""
    name: str                        # display name (usually the run dir)
    source: Path                     # run directory, relative to project root
    family: str                      # 'yolo' | 'rfdetr' | 'unknown'
    files: list[ONNXFile] = field(default_factory=list)

    @property
    def total_size_mb(self) -> float:
        return sum(f.size_mb for f in self.files)


def _classify_variant(path: Path) -> str:
    """Derive a compound variant label from the filename.

    Examples:
        best.onnx                → 'base'
        best.sim.onnx            → 'sim'
        best.fp16.onnx           → 'fp16'
        best.fp16.sim.onnx       → 'fp16+sim'
        best.sim.fp16.onnx       → 'sim+fp16'
        inference_model.onnx     → 'rfdetr-export'
        inference_model.sim.onnx → 'rfdetr-export+sim'
    """
    name = path.name.lower()
    stem = name.rsplit(".onnx", 1)[0]
    parts = stem.split(".") if stem else []

    # First segment: model-family export hint (e.g. 'inference_model' from RF-DETR).
    initial_tag: str | None = None
    if parts and parts[0] == "inference_model":
        initial_tag = "rfdetr-export"

    # Subsequent segments: compression/simplification tags, in filename order.
    extras: list[str] = []
    for tok in parts[1:]:
        tag = _TOKEN_TAG.get(tok)
        if tag and tag not in extras:
            extras.append(tag)

    if initial_tag and extras:
        return initial_tag + "+" + "+".join(extras)
    if initial_tag:
        return initial_tag
    if extras:
        return "+".join(extras)
    return "base"


def _detect_family(path: Path) -> str:
    """Walk the path's ancestors; return the first recognised family hint."""
    for part in path.parts:
        p = part.lower()
        if p in _FAMILY_HINTS:
            return _FAMILY_HINTS[p]
    return "unknown"


def _pick_run_dir(onnx_path: Path) -> Path:
    """Pick the 'run directory' that should represent this file.

    Heuristic: if the immediate parent is ``weights/``, use the
    grandparent (Ultralytics convention). Otherwise use the direct
    parent (RF-DETR convention, etc).
    """
    parent = onnx_path.parent
    if parent.name == "weights":
        return parent.parent
    return parent


def _should_skip(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    for part in rel.parts:
        if part in _SKIP_DIRS:
            return True
    return False


def discover_onnx(root: Path) -> list[ModelGroup]:
    """Scan ``root`` recursively for ``*.onnx`` files, return grouped."""
    root = root.resolve()
    groups: dict[Path, ModelGroup] = {}

    for onnx in root.rglob("*.onnx"):
        if _should_skip(onnx, root):
            continue

        run_dir = _pick_run_dir(onnx)
        try:
            run_rel = run_dir.relative_to(root)
        except ValueError:
            continue  # shouldn't happen but be defensive

        if run_rel not in groups:
            groups[run_rel] = ModelGroup(
                name=run_dir.name,
                source=run_rel,
                family=_detect_family(onnx),
            )

        size_mb = round(onnx.stat().st_size / (1024 * 1024), 2)
        groups[run_rel].files.append(
            ONNXFile(
                path=onnx,
                rel=onnx.relative_to(root),
                size_mb=size_mb,
                variant=_classify_variant(onnx),
            )
        )

    # Stable sort: family first, then run name. "unknown" last.
    def _sort_key(g: ModelGroup):
        order = {"yolo": 0, "rfdetr": 1}.get(g.family, 99)
        return (order, g.name.lower())

    # Sort files within each group by variant then name
    for g in groups.values():
        g.files.sort(key=lambda f: (f.variant, f.path.name))

    return sorted(groups.values(), key=_sort_key)
