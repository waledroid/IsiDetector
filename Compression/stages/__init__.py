"""Compression stage registry.

Each compression strategy (FP16, INT8, TensorRT, ...) will live in
its own module under this package and register itself with
``@register``. v1 ships only the base contract; v2+ adds real stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Stage

#: The stage registry. Populated by ``@register`` decorators as modules import.
STAGES: dict[str, type["Stage"]] = {}


def register(cls):
    """Decorator: register a Stage subclass under its ``name`` attribute."""
    STAGES[cls.name] = cls
    return cls


# ── Auto-register built-in stages ──────────────────────────────────────────
# Importing each stage module is what fires its ``@register`` decorator.
# Deferred imports live at the bottom so the ``register`` symbol above is
# defined before any stage module tries to use it.

from . import fp16      # noqa: E402, F401  — registers FP16Stage
from . import int8      # noqa: E402, F401  — registers INT8Stage (dynamic, CPU-only)
from . import int8_qdq  # noqa: E402, F401  — registers INT8QDQStage (static, cross-EP)
from . import sim       # noqa: E402, F401  — registers SimplifyStage
