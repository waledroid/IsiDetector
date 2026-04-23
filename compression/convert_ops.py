"""Format conversion + OpenVINO FP16 compression helpers.

Wraps the existing ``src.inference.export_engine`` pipeline with the
same phase-print + heartbeat UX that the ONNX compression stages use.
Every long-running call prints a "⏳ …" line before it fires, then a
"✓ …" line with elapsed seconds after. Calls that might stall for tens
of seconds are wrapped in :func:`Compression.stages.fp16._heartbeat`
so the operator sees "still <phase> (Xs elapsed)" every 10 s instead
of a silent spinner.

Operations exposed:

* :func:`pt_to_onnx`           — ``.pt`` (YOLO) or ``.pth`` (RF-DETR) → ``.onnx``
* :func:`onnx_to_sim`          — ``.onnx`` → ``.sim.onnx`` (onnxsim)
* :func:`onnx_to_openvino`     — ``.onnx`` → OpenVINO IR (``.xml`` + ``.bin``)
* :func:`openvino_fp16`        — FP32 ``.xml`` → FP16 ``.xml`` (weight compression)
* :func:`pt_full_pipeline`     — ``.pt`` → ``.onnx`` → ``.sim.onnx`` → OpenVINO IR

Used by both the interactive "convert" menu (``cli.py``) and the
one-shot CLI (``__main__.py --convert MODE``).
"""
from __future__ import annotations

import time
from pathlib import Path

from .stages.fp16 import _heartbeat
from .ui import console


# ── .pt / .pth → .onnx ──────────────────────────────────────────────────────

def pt_to_onnx(pt_path: Path, imgsz: int | None = None) -> Path:
    """Export a native weights file to ONNX.

    Dispatches on file extension via ``export_from_weights``:
    ``.pt`` → Ultralytics YOLO exporter, ``.pth`` → RF-DETR exporter.
    """
    from src.inference.export_engine import export_from_weights

    pt_path = Path(pt_path)
    size_mb = pt_path.stat().st_size / (1024 * 1024)
    console.print(
        f"   [dim]⏳ exporting {pt_path.name} ({size_mb:.0f} MB) to ONNX…[/dim]"
    )
    t0 = time.perf_counter()
    with _heartbeat(f"exporting {pt_path.suffix} → ONNX"):
        onnx_path = export_from_weights(pt_path, imgsz=imgsz)
    console.print(f"   [dim]✓ exported in {time.perf_counter() - t0:.1f}s[/dim]")
    return Path(onnx_path)


# ── .onnx → .sim.onnx ───────────────────────────────────────────────────────

def onnx_to_sim(onnx_path: Path) -> Path:
    """Run onnxsim on an ONNX file.

    Reuses :func:`src.inference.export_engine.optimize_onnx`, which
    is idempotent — passing an already-simplified file returns it
    unchanged without re-running simplification.
    """
    from src.inference.export_engine import optimize_onnx

    onnx_path = Path(onnx_path)
    console.print(f"   [dim]⏳ simplifying {onnx_path.name}…[/dim]")
    t0 = time.perf_counter()
    with _heartbeat("simplifying"):
        sim_path = optimize_onnx(onnx_path)
    console.print(f"   [dim]✓ simplified in {time.perf_counter() - t0:.1f}s[/dim]")
    return Path(sim_path)


# ── .onnx → OpenVINO IR ─────────────────────────────────────────────────────

def onnx_to_openvino(onnx_path: Path, output_dir: Path | None = None) -> Path:
    """Convert an ONNX file to OpenVINO IR (``.xml`` + ``.bin``).

    In OpenVINO 2024+ ``ov.save_model`` defaults to
    ``compress_to_fp16=True``, so the produced IR is already FP16-
    compressed unless the runtime is pre-2024. For explicit FP32→FP16
    compression of an existing IR see :func:`openvino_fp16`.
    """
    from src.inference.export_engine import convert_openvino

    onnx_path = Path(onnx_path)
    console.print(
        f"   [dim]⏳ converting {onnx_path.name} → OpenVINO IR…[/dim]"
    )
    t0 = time.perf_counter()
    with _heartbeat("converting to OpenVINO"):
        xml_path = convert_openvino(onnx_path, output_dir=output_dir)
    console.print(f"   [dim]✓ converted in {time.perf_counter() - t0:.1f}s[/dim]")
    return Path(xml_path)


# ── OpenVINO FP32 IR → OpenVINO FP16 IR ─────────────────────────────────────

def openvino_fp16(xml_path: Path, output_path: Path | None = None) -> Path:
    """Re-save an OpenVINO IR with ``compress_to_fp16=True``.

    Input is a ``.xml`` whose corresponding ``.bin`` holds FP32 weights;
    output is a new ``.xml`` + ``.bin`` pair with FP16 weights (roughly
    half the .bin size).

    Output defaults to ``<stem>.fp16.xml`` next to the source.
    """
    import openvino as ov

    xml_path = Path(xml_path)
    if xml_path.suffix.lower() != ".xml":
        raise ValueError(
            f"openvino_fp16 expects an OpenVINO .xml file, got {xml_path.suffix}"
        )
    if output_path is None:
        stem = xml_path.name.rsplit(".xml", 1)[0]
        output_path = xml_path.parent / f"{stem}.fp16.xml"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Bin file holds the weights — that's the useful size metric.
    bin_path = xml_path.with_suffix(".bin")
    bin_mb = bin_path.stat().st_size / (1024 * 1024) if bin_path.exists() else 0.0

    console.print(
        f"   [dim]⏳ loading {xml_path.name} "
        f"(weights: {bin_mb:.0f} MB)…[/dim]"
    )
    t0 = time.perf_counter()
    model = ov.Core().read_model(str(xml_path))
    console.print(f"   [dim]✓ loaded in {time.perf_counter() - t0:.1f}s[/dim]")

    console.print("   [dim]⏳ re-saving with compress_to_fp16=True…[/dim]")
    t0 = time.perf_counter()
    with _heartbeat("compressing to FP16"):
        ov.save_model(model, str(output_path), compress_to_fp16=True)
    console.print(f"   [dim]✓ saved in {time.perf_counter() - t0:.1f}s[/dim]")

    new_bin = output_path.with_suffix(".bin")
    new_bin_mb = new_bin.stat().st_size / (1024 * 1024) if new_bin.exists() else 0.0
    if bin_mb and new_bin_mb:
        delta_pct = (new_bin_mb / bin_mb - 1) * 100
        verb = "grew" if delta_pct > 0 else "shrank"
        colour = "yellow" if delta_pct > 0 else "green"
        console.print(
            f"   [dim].bin: {bin_mb:.2f} → {new_bin_mb:.2f} MB "
            f"([{colour}]{verb} {abs(delta_pct):.1f}%[/{colour}])[/dim]"
        )

    return output_path


# ── Pipeline: .pt → .onnx → .sim.onnx → OpenVINO IR ─────────────────────────

def pt_full_pipeline(
    pt_path: Path,
    imgsz: int | None = None,
) -> dict[str, Path]:
    """Run the full deploy-pipeline: ``.pt`` → ONNX → sim → OpenVINO IR.

    Returns a dict with keys ``onnx``, ``sim``, ``openvino`` mapping to
    the produced paths. Useful both interactively (one menu pick runs
    the whole pipeline) and from CI scripts.
    """
    results: dict[str, Path] = {}

    console.print("\n[bold]Step 1/3: export to ONNX[/bold]")
    results["onnx"] = pt_to_onnx(pt_path, imgsz=imgsz)

    console.print("\n[bold]Step 2/3: simplify with onnxsim[/bold]")
    results["sim"] = onnx_to_sim(results["onnx"])

    console.print("\n[bold]Step 3/3: convert to OpenVINO IR[/bold]")
    results["openvino"] = onnx_to_openvino(results["sim"])

    return results
