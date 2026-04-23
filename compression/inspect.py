"""ONNX introspection — pull internal properties without running the model.

Used by the CLI to show operators what they're about to compress, and to
detect whether a given transformation has already been applied (so the
stage picker can disable stages that would be a no-op or redundant).

Single read of the file. Does NOT load external weight blobs —
``onnx.load(..., load_external_data=False)`` so this stays fast even on
large models with side-car tensor files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx  # runtime dep declared in compression/requirements.txt


#: ONNX TensorProto.DataType enum → friendly dtype name.
#: Mirrored from onnx/onnx/onnx.in.proto → TensorProto.DataType.
_TYPE_MAP: dict[int, str] = {
    0: "undefined",
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}

_QUANTIZE_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}


@dataclass
class TensorSpec:
    """One input/output tensor — name, dtype, and shape."""
    name: str
    dtype: str                              # e.g. 'float32', 'float16'
    shape: list[Any]                        # list of int or str (for symbolic dims)


@dataclass
class ONNXProperties:
    """Everything the CLI needs to know about an ONNX file at a glance."""

    # File metadata
    path: Path
    size_mb: float

    # Graph metadata
    opset_version: int
    ir_version: int
    producer_name: str
    producer_version: str

    # I/O signatures
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)

    # Precision analysis
    weight_dtype: str = "unknown"           # dominant initializer dtype
    param_count: int = 0                    # approx — sum of initializer elements

    # Graph structure — the op histogram. Useful for diffing source
    # vs compressed output (e.g. "INT8 added 24 QuantizeLinear nodes").
    op_counts: dict[str, int] = field(default_factory=dict)
    total_nodes: int = 0

    # Structural flags
    has_quantize_nodes: bool = False
    has_dynamic_axes: bool = False

    # "Already applied" flags (drive stage-picker disables)
    already_fp16: bool = False
    already_int8: bool = False
    already_simplified: bool = False        # filename hint — onnxsim doesn't stamp the graph


# ── Helpers ─────────────────────────────────────────────────────────────────

def _tensor_dim(dim) -> Any:
    """Return concrete int for fixed dims, symbol string for dynamic, '?' for empty."""
    if dim.dim_value:
        return dim.dim_value
    if dim.dim_param:
        return dim.dim_param
    return "?"


def _parse_value_info(vi) -> TensorSpec:
    """Convert an onnx.ValueInfoProto into a TensorSpec."""
    tt = vi.type.tensor_type
    return TensorSpec(
        name=vi.name,
        dtype=_TYPE_MAP.get(tt.elem_type, "unknown"),
        shape=[_tensor_dim(d) for d in tt.shape.dim],
    )


def _analyse_weights(graph) -> tuple[str, int]:
    """Return (dominant_dtype, total_parameter_count)."""
    counts: dict[str, int] = {}
    total = 0
    for init in graph.initializer:
        dt = _TYPE_MAP.get(init.data_type, "unknown")
        size = 1
        for d in init.dims:
            size *= int(d) if d else 1
        total += size
        counts[dt] = counts.get(dt, 0) + size

    if not counts:
        return "unknown", 0

    dominant = max(counts, key=counts.get)
    # If no single dtype holds ≥90 % of params, call it mixed.
    top_share = counts[dominant] / sum(counts.values())
    if top_share < 0.9 and len(counts) > 1:
        return "mixed", total
    return dominant, total


def _has_quantize_nodes(graph) -> bool:
    return any(n.op_type in _QUANTIZE_OPS for n in graph.node)


def _has_dynamic_axes(graph) -> bool:
    for inp in graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            # Symbolic param (e.g. 'N', 'batch') with no fixed value → dynamic
            if dim.dim_param and not dim.dim_value:
                return True
    return False


def _real_inputs(graph) -> list:
    """Graph inputs excluding the ones that are actually initializers.

    Older ONNX exports include every initializer in ``graph.input`` as well,
    so filtering them out gives the true runtime-fed inputs.
    """
    init_names = {init.name for init in graph.initializer}
    return [inp for inp in graph.input if inp.name not in init_names]


# ── Public entry point ──────────────────────────────────────────────────────

def inspect_onnx(path: Path) -> ONNXProperties:
    """Read ``path`` as ONNX and build a properties summary."""
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    weight_dtype, param_count = _analyse_weights(graph)

    # Opset: the default ('') or 'ai.onnx' domain is what drives dtype support.
    opset_version = max(
        (imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")),
        default=0,
    )

    has_q = _has_quantize_nodes(graph)

    # Op histogram — Counter-of-op_type → count. Drives the "what changed"
    # delta in the post-stage summary panel.
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    return ONNXProperties(
        path=path,
        size_mb=size_mb,
        opset_version=opset_version,
        ir_version=model.ir_version,
        producer_name=model.producer_name or "unknown",
        producer_version=model.producer_version or "",
        inputs=[_parse_value_info(i) for i in _real_inputs(graph)],
        outputs=[_parse_value_info(o) for o in graph.output],
        weight_dtype=weight_dtype,
        param_count=param_count,
        op_counts=op_counts,
        total_nodes=sum(op_counts.values()),
        has_quantize_nodes=has_q,
        has_dynamic_axes=_has_dynamic_axes(graph),
        already_fp16=(weight_dtype == "float16"),
        already_int8=(weight_dtype == "int8" or has_q),
        already_simplified=".sim." in path.name,
    )
