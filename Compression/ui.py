"""Shared UI widgets for the Compression CLI.

All rendering flows through a single ``rich.Console`` so output is
consistent across modules.
"""

from __future__ import annotations

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __title__, __version__
from .benchmark import BenchmarkResult
from .discovery import ModelGroup
from .inspect import ONNXProperties, TensorSpec
from .validate import ValidationResult

#: Shared console. Every print in the tool goes through this.
console = Console()

#: What the tool can / will do. Emoji + short name + one-line pitch.
CAPABILITIES: list[tuple[str, str, str]] = [
    ("🪶", "FP16 conversion",     "Halve model size with ~no accuracy loss"),
    ("🔢", "INT8 quantisation",   "Quarter size, ~1–3 % mAP drop, calibration-based"),
    ("⚡", "TensorRT compile",    "GPU-specific engine, 1.5–3× faster than ONNX-CUDA"),
    ("📊", "Benchmark",           "Compare size + FPS across all variants of a model"),
    ("🧪", "Accuracy validate",   "Diff predictions against the baseline on a test frame"),
]


def print_banner() -> None:
    """Render the title banner."""
    body = Text.assemble(
        ("🔧  ", "bold cyan"),
        (f"{__title__}", "bold white"),
        ("   ", ""),
        (f"v{__version__}", "dim cyan"),
        ("\n", ""),
        ("\n", ""),
        ("Shrink your ONNX models for faster, smaller production inference.\n", "white"),
        ("Pick a strategy, feed it a model, ship a smaller artefact.", "dim"),
    )
    console.print(
        Panel(
            Align.left(body),
            box=box.HEAVY,
            border_style="cyan",
            padding=(1, 3),
        )
    )


def print_capabilities() -> None:
    """Render the feature list below the banner."""
    console.print("\n[bold]✨ What this tool can do:[/bold]\n")
    for emoji, name, desc in CAPABILITIES:
        console.print(f"   {emoji}  [bold]{name:<22}[/bold] [dim]{desc}[/dim]")
    console.print()


def build_models_table(groups: list[ModelGroup]) -> Table:
    """Render the discovered ONNX groups as a nicely bordered table."""
    table = Table(
        title="📂 Detected ONNX Models",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title_justify="left",
        title_style="bold",
    )
    table.add_column("#",        style="dim",  width=3, justify="right")
    table.add_column("Family",   width=8)
    table.add_column("Run",      style="cyan", no_wrap=False)
    table.add_column("Files",    justify="right", width=5)
    table.add_column("Size",     justify="right", style="green", width=10)
    table.add_column("Variants", style="yellow")
    table.add_column("Source",   style="dim")

    if not groups:
        return table  # caller can detect empty via row_count

    for i, g in enumerate(groups, 1):
        variants = ", ".join(sorted({f.variant for f in g.files}))
        table.add_row(
            str(i),
            _family_emoji(g.family) + " " + g.family,
            g.name,
            str(len(g.files)),
            f"{g.total_size_mb:.1f} MB",
            variants,
            str(g.source),
        )
    return table


def _family_emoji(family: str) -> str:
    return {"yolo": "🟢", "rfdetr": "🟣"}.get(family, "⚪")


def print_group_detail(group: ModelGroup) -> None:
    """Expand one group into a per-file table. Used when the user picks a model."""
    console.print()
    console.print(
        Panel(
            f"[bold]{_family_emoji(group.family)} {group.name}[/bold]   "
            f"[dim]{group.family}[/dim]\n"
            f"[dim]{group.source}[/dim]",
            border_style="cyan",
            box=box.SIMPLE,
        )
    )

    table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Variant", style="yellow")
    table.add_column("File")
    table.add_column("Size", justify="right", style="green")

    for i, f in enumerate(group.files, 1):
        table.add_row(str(i), f.variant, f.path.name, f"{f.size_mb:.2f} MB")
    console.print(table)


# ── Properties panel ────────────────────────────────────────────────────────

_WEIGHT_DTYPE_STYLE = {
    "float32": "white",
    "float16": "cyan",
    "int8":    "magenta",
    "uint8":   "magenta",
    "mixed":   "yellow",
}


def _shape_to_str(shape) -> str:
    """Render a TensorSpec.shape (mix of ints and strings) as '[1, 3, 416, 416]'."""
    return "[" + ", ".join(str(d) for d in shape) + "]"


def _tensor_line(idx: int, role: str, spec: TensorSpec) -> str:
    """One row: 'Input[0] .... images    [1, 3, 416, 416]  float32'."""
    name = f"{spec.name:<14}"
    shape = _shape_to_str(spec.shape)
    return f"   {role}[{idx}] ... [cyan]{name}[/] {shape:<26} [dim]{spec.dtype}[/dim]"


def _applied_row(label: str, applied: bool, reason_on: str, reason_off: str) -> str:
    """Render an ● / ○ row for the 'Applied transformations' section."""
    marker = "[green]●[/]" if applied else "[dim]○[/]"
    reason = reason_on if applied else reason_off
    label_colour = "bold" if applied else "dim"
    return f"   {marker} [{label_colour}]{label:<11}[/{label_colour}] [dim]({reason})[/dim]"


def print_properties(props: ONNXProperties) -> None:
    """Render the per-ONNX properties summary panel."""
    lines: list[str] = []

    # ── File ──
    lines.append("[bold]📄 File[/bold]")
    lines.append(f"   Path ........ [dim]{props.path}[/dim]")
    lines.append(f"   Size ........ {props.size_mb:.2f} MB")
    producer = f"{props.producer_name}"
    if props.producer_version:
        producer += f" {props.producer_version}"
    lines.append(f"   Producer .... {producer}")
    lines.append(f"   Opset ....... {props.opset_version}    IR version: {props.ir_version}")
    lines.append("")

    # ── Precision ──
    lines.append("[bold]🧮 Precision[/bold]")
    dt_style = _WEIGHT_DTYPE_STYLE.get(props.weight_dtype, "red")
    lines.append(f"   Weights ..... [{dt_style}]{props.weight_dtype}[/{dt_style}]")
    if props.inputs:
        lines.append(f"   Input dtype . [dim]{props.inputs[0].dtype}[/dim]")
    if props.param_count:
        approx = (
            f"{props.param_count / 1_000_000:.1f}M"
            if props.param_count >= 1_000_000
            else f"{props.param_count / 1_000:.0f}K"
        )
        lines.append(f"   Approx params {approx}")
    if props.has_dynamic_axes:
        lines.append("   Shape ....... [yellow]dynamic input dims detected[/yellow]")
    lines.append("")

    # ── I/O ──
    lines.append("[bold]🔢 I/O[/bold]")
    for i, inp in enumerate(props.inputs):
        lines.append(_tensor_line(i, "Input ", inp))
    for i, out in enumerate(props.outputs):
        lines.append(_tensor_line(i, "Output", out))
    lines.append("")

    # ── Graph (op histogram, top 8) ──
    if props.op_counts:
        top = sorted(props.op_counts.items(), key=lambda kv: -kv[1])[:8]
        lines.append("[bold]🧬 Graph[/bold]")
        lines.append(f"   Total nodes ... {props.total_nodes}")
        lines.append(f"   Op types ...... {len(props.op_counts)}")
        lines.append(f"   Top ops ....... " + ", ".join(
            f"[cyan]{op}[/cyan] [dim]{n}[/dim]" for op, n in top
        ))
        lines.append("")

    # ── Applied transformations ──
    lines.append("[bold]✓ Applied transformations[/bold]")
    lines.append(_applied_row(
        "FP16", props.already_fp16,
        reason_on="weights are float16 — FP16 stage would be a no-op",
        reason_off="weights are not float16",
    ))
    lines.append(_applied_row(
        "INT8", props.already_int8,
        reason_on=(
            "QuantizeLinear nodes present" if props.has_quantize_nodes
            else "weight dtype is int8"
        ),
        reason_off="no quantise nodes found",
    ))
    lines.append(_applied_row(
        "Simplified", props.already_simplified,
        reason_on="filename contains '.sim'",
        reason_off="not marked as simplified",
    ))

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title=f"🔍 ONNX Properties — {props.path.name}",
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def print_output_summary(out_props: ONNXProperties, src_props: ONNXProperties) -> None:
    """Compact summary panel shown after a stage writes a new artefact.

    Includes deltas vs the source so the operator can see what the
    transformation actually changed (size, weight dtype, new ops
    introduced like QuantizeLinear / DequantizeLinear, total node
    count drift).
    """
    lines: list[str] = []

    # ── New artefact ──
    lines.append("[bold]📦 New artefact[/bold]")
    lines.append(f"   Path .......... [dim]{out_props.path}[/dim]")
    if src_props.size_mb > 0:
        delta_pct = (out_props.size_mb / src_props.size_mb - 1) * 100
        verb_colour = "green" if delta_pct < 0 else "yellow"
        lines.append(
            f"   Size .......... {out_props.size_mb:.2f} MB   "
            f"[dim](source {src_props.size_mb:.2f} MB, "
            f"[{verb_colour}]{delta_pct:+.1f}%[/{verb_colour}])[/dim]"
        )
    else:
        lines.append(f"   Size .......... {out_props.size_mb:.2f} MB")

    # ── Precision change ──
    lines.append("")
    lines.append("[bold]🧮 Precision[/bold]")
    if out_props.weight_dtype != src_props.weight_dtype:
        lines.append(
            f"   Weights ....... [cyan]{out_props.weight_dtype}[/cyan]   "
            f"[dim](was {src_props.weight_dtype})[/dim]"
        )
    else:
        lines.append(f"   Weights ....... {out_props.weight_dtype}   [dim](unchanged)[/dim]")
    if out_props.param_count:
        approx = (
            f"{out_props.param_count / 1_000_000:.1f}M"
            if out_props.param_count >= 1_000_000
            else f"{out_props.param_count / 1_000:.0f}K"
        )
        lines.append(f"   Approx params  {approx}")

    # ── Graph diff ──
    lines.append("")
    lines.append("[bold]🧬 Graph[/bold]")
    node_delta = out_props.total_nodes - src_props.total_nodes
    sign = "+" if node_delta >= 0 else ""
    delta_colour = "yellow" if node_delta != 0 else "dim"
    lines.append(
        f"   Total nodes ... {out_props.total_nodes}   "
        f"[dim](source {src_props.total_nodes}, "
        f"[{delta_colour}]{sign}{node_delta}[/{delta_colour}])[/dim]"
    )

    # New op types that didn't exist in source — shows what the stage
    # actually inserted (QuantizeLinear after INT8, etc.)
    new_op_types = set(out_props.op_counts) - set(src_props.op_counts)
    if new_op_types:
        lines.append(f"   New op types .. " + ", ".join(
            f"[bold green]{op}[/bold green] [dim]{out_props.op_counts[op]}[/dim]"
            for op in sorted(new_op_types)
        ))

    # Op types whose count changed materially (>1)
    diffs: list[tuple[str, int]] = []
    for op, n in out_props.op_counts.items():
        d = n - src_props.op_counts.get(op, 0)
        if d != 0 and op not in new_op_types:
            diffs.append((op, d))
    diffs.sort(key=lambda kv: -abs(kv[1]))
    if diffs:
        diff_str = ", ".join(
            f"[cyan]{op}[/cyan] [{('green' if d < 0 else 'yellow')}]{d:+d}[/]"
            for op, d in diffs[:6]
        )
        lines.append(f"   Op count Δ .... {diff_str}")

    # Top ops always shown so operator gets a sense of the new graph shape
    top = sorted(out_props.op_counts.items(), key=lambda kv: -kv[1])[:6]
    if top:
        lines.append(f"   Top ops ....... " + ", ".join(
            f"[cyan]{op}[/cyan] [dim]{n}[/dim]" for op, n in top
        ))

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title=f"📊 Output summary — {out_props.path.name}",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


# ── Benchmark results ──────────────────────────────────────────────────────

def print_benchmark_results(
    group_name: str,
    results: list[BenchmarkResult],
    sample_path,
    iterations: int,
) -> None:
    """Render the benchmark table + a subtitle noting test conditions."""

    # Find the baseline (variant == 'base' or the first non-errored row)
    base = next(
        (r for r in results if r.file.variant == "base" and r.error is None),
        None,
    )
    if base is None:
        base = next((r for r in results if r.error is None), None)

    # Fastest non-errored row gets the medal.
    ok = [r for r in results if r.error is None and r.fps > 0]
    fastest = max(ok, key=lambda r: r.fps) if ok else None

    table = Table(
        title=f"📊 Benchmark — {group_name}",
        title_justify="left",
        title_style="bold",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Variant",    style="cyan",   no_wrap=True)
    table.add_column("File",       style="dim")
    table.add_column("Size",       justify="right", style="green")
    table.add_column("Median ms",  justify="right")
    table.add_column("p95 ms",     justify="right")
    table.add_column("FPS",        justify="right", style="bold")
    table.add_column("vs base",    justify="right")
    table.add_column("Provider",   style="dim")

    for r in results:
        # vs-base column
        if r.error is not None:
            vs = "[red]error[/red]"
        elif r is base:
            vs = "[dim]—[/dim]"
        elif base is not None and base.fps > 0:
            fps_delta = (r.fps / base.fps - 1) * 100
            size_delta = (r.size_mb / base.size_mb - 1) * 100
            fps_colour = "green" if fps_delta > 0 else "yellow" if fps_delta > -5 else "red"
            vs = (
                f"[{fps_colour}]{fps_delta:+.0f}% FPS[/{fps_colour}] "
                f"[dim]{size_delta:+.0f}% size[/dim]"
            )
        else:
            vs = ""

        if r.error is not None:
            row_style = "dim"
            table.add_row(
                r.file.variant,
                r.file.path.name,
                f"{r.size_mb:.2f} MB",
                "-",
                "-",
                "-",
                vs,
                "-",
                style=row_style,
            )
            continue

        fps_txt = f"{r.fps:6.1f}"
        if fastest is not None and r is fastest:
            fps_txt = f"[bold green]🥇 {r.fps:6.1f}[/bold green]"

        table.add_row(
            r.file.variant,
            r.file.path.name,
            f"{r.size_mb:.2f} MB",
            f"{r.median_ms:6.2f}",
            f"{r.p95_ms:6.2f}",
            fps_txt,
            vs,
            r.provider.replace("ExecutionProvider", ""),
        )

    console.print()
    console.print(table)

    # Subtitle — where the sample frame came from and how many iters we did.
    src = str(sample_path) if sample_path else "grey 720×1280 dummy frame"
    console.print(
        f"[dim]   Sample: {src}   │   "
        f"iterations: {iterations}   │   "
        f"warmup discarded[/dim]\n"
    )

    # Print per-variant error reasons below the table for quick triage.
    for r in results:
        if r.error is not None:
            console.print(
                f"[red]   ❌  {r.file.variant}:[/red] [dim]{r.error}[/dim]"
            )
    if any(r.error for r in results):
        console.print()


# ── Validation results ─────────────────────────────────────────────────────

_VERDICT_STYLE = {
    "good":         ("✅", "bold green",   "GOOD"),
    "acceptable":   ("✅", "bold green",   "ACCEPTABLE"),
    "degraded":     ("⚠️",  "bold yellow",  "DEGRADED"),
    "broken":       ("❌", "bold red",     "BROKEN"),
    "no-baseline":  ("❓", "bold yellow",  "NO BASELINE DATA"),
}


def print_validation_result(result: ValidationResult) -> None:
    """Render a validation pair diff as a single bordered panel."""
    emoji, style, label = _VERDICT_STYLE.get(
        result.verdict, ("❓", "bold", result.verdict.upper())
    )

    lines: list[str] = []

    # ── Header ──
    lines.append("[bold]🎯 Pair[/bold]")
    lines.append(f"   Baseline  ....... [cyan]{result.baseline_file.path.name}[/cyan] "
                 f"[dim]({result.baseline_file.variant}, {result.baseline_file.size_mb:.2f} MB)[/dim]")
    lines.append(f"   Candidate ....... [cyan]{result.candidate_file.path.name}[/cyan] "
                 f"[dim]({result.candidate_file.variant}, {result.candidate_file.size_mb:.2f} MB)[/dim]")
    lines.append("")

    # ── Sample set ──
    lines.append("[bold]🖼  Sample set[/bold]")
    lines.append(f"   Frames checked .. {result.n_frames_checked}")
    if result.sample_dir is not None:
        lines.append(f"   Source .......... [dim]{result.sample_dir}[/dim]")
    lines.append("")

    # ── Counts ──
    lines.append("[bold]🔢 Detection counts[/bold]")
    lines.append(f"   Baseline ........ {result.baseline_detections}")
    diff = result.candidate_detections - result.baseline_detections
    diff_txt = f"{diff:+d}"
    keep_pct = result.keep_rate * 100
    lines.append(f"   Candidate ....... {result.candidate_detections} "
                 f"[dim]({diff_txt}, {keep_pct:.1f}% kept)[/dim]")
    lines.append("")

    # ── Matched pairs ──
    lines.append("[bold]🔗 Matching (IoU ≥ " f"{result.iou_threshold:.2f}" ")[/bold]")
    lines.append(f"   Matched pairs ... {result.matched_pairs} "
                 f"[dim]({result.match_rate * 100:.1f}% of baseline)[/dim]")
    lines.append(f"   Class agreement . {result.class_match_rate * 100:.1f}%")
    if result.matched_pairs > 0:
        lines.append(f"   Mean box drift .. {result.mean_box_drift_px:.1f} px")
        lines.append(f"   Max  box drift .. {result.max_box_drift_px:.1f} px")
        sign = "less confident" if result.mean_conf_delta > 0 else "more confident"
        lines.append(f"   Mean conf Δ ..... {result.mean_conf_delta:+.3f}  "
                     f"[dim]({sign})[/dim]")
    lines.append("")

    # ── Verdict ──
    lines.append(f"[bold]{emoji} Verdict[/bold]  "
                 f"[{style}]{label}[/{style}]  "
                 f"[dim]{result.verdict_reason}[/dim]")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title=f"🧪 Validation — {result.candidate_file.variant} vs {result.baseline_file.variant}",
            title_align="left",
            border_style=style.split()[-1],   # take the colour from the verdict style
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
