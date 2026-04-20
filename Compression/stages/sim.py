"""ONNX graph simplification stage.

Runs ``onnxsim`` — constant folding, dead-op removal, shape-inference
cleanup. Not strictly "compression" in the weight-reduction sense, but
it shrinks the graph (fewer nodes, smaller metadata) and can unblock
some later stages (INT8 quantisation of constants, TensorRT parsers
that reject redundant ops).

Idempotent: running it twice on the same file would produce an
identical artefact, so ``can_run`` refuses if the filename already
carries the ``.sim.`` tag.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import onnxsim

from ..inspect import ONNXProperties
from . import register
from .base import Stage


@register
class SimplifyStage(Stage):
    name = "sim"
    emoji = "✨"
    description = "Run onnxsim — fold constants, drop redundant ops, clean up shape info"

    def can_run(self, props: ONNXProperties) -> tuple[bool, str]:
        if props.already_simplified:
            return False, (
                "file name already contains '.sim' — running again would "
                "just produce an identical graph"
            )
        return True, ""

    def run(self, src: Path) -> Path:
        out = self.output_path(src)

        model = onnx.load(str(src))
        simplified, check_ok = onnxsim.simplify(model)

        if not check_ok:
            # onnxsim runs a post-simplify shape-check on the whole graph;
            # failure means the output isn't guaranteed equivalent to the input.
            # Better to abort than silently ship a broken model.
            raise RuntimeError(
                "onnxsim verification pass failed — the simplified graph did "
                "not match the original. Output not written."
            )

        onnx.save(simplified, str(out))
        return out
