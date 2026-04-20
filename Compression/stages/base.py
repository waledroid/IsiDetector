"""Abstract base class every compression stage must inherit from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..inspect import ONNXProperties


class Stage(ABC):
    """Contract for a single compression step.

    Concrete stages (FP16, INT8, TensorRT, ...) live under
    ``Compression/stages/<name>.py`` and register themselves via the
    ``@register`` decorator from ``Compression.stages.__init__``.

    The CLI looks a stage up by ``name``, calls ``can_run(props)`` for
    a pre-flight check, asks ``output_path(src)`` where the artefact
    will land (so it can prompt before overwriting), and then calls
    ``run(src)`` to perform the conversion.
    """

    #: Registry key — short, lowercase, dash-free. Used in menus and file suffixes.
    name: str = ""

    #: Emoji shown next to the stage in the UI.
    emoji: str = "•"

    #: One-line pitch shown next to the stage.
    description: str = ""

    def can_run(self, props: ONNXProperties) -> tuple[bool, str]:
        """Pre-flight check.

        Return ``(True, "")`` if the stage is applicable, or
        ``(False, reason)`` if not (e.g. "already FP16", "needs GPU").
        Default implementation allows everything — subclasses override
        for stage-specific guards.
        """
        return True, ""

    def output_path(self, src: Path) -> Path:
        """Where this stage writes its artefact.

        Default: sibling file named ``<stem>.<stage_name>.onnx`` in
        the same directory as the source. So ``best.onnx`` with stage
        ``fp16`` produces ``best.fp16.onnx`` next to it. Subclasses
        can override (e.g. TensorRT writes ``.engine`` with hardware
        tags in the filename).
        """
        stem = src.name.rsplit(".onnx", 1)[0]
        return src.parent / f"{stem}.{self.name}.onnx"

    @abstractmethod
    def run(self, src: Path) -> Path:
        """Execute the compression. Return the absolute output path.

        Implementations should:
        - write exactly one artefact file
        - not mutate the source
        - raise a clear exception on failure (caller renders it)
        """
