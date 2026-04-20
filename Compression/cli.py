"""Isitec Model Compression Tool — interactive entry point.

Run with:
    python -m Compression

Or via the root wrapper:
    ./compress.sh
"""

from __future__ import annotations

from pathlib import Path

import questionary
from questionary import Choice, Style

from .benchmark import run_group_benchmark
from .discovery import ModelGroup, ONNXFile, discover_onnx
from .inspect import ONNXProperties, inspect_onnx
from .stages import STAGES  # noqa: F401 — imports trigger @register
from .ui import (
    build_models_table,
    console,
    print_banner,
    print_benchmark_results,
    print_capabilities,
    print_group_detail,
    print_output_summary,
    print_properties,
    print_validation_result,
)
from .validate import validate_pair

#: The project root is two levels up from this file
#: (.../logistic/Compression/cli.py → .../logistic).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Visual theme for questionary — matches the rich banner palette.
_Q_STYLE = Style(
    [
        ("qmark",       "fg:#00d7ff bold"),      # leading ? mark
        ("question",    "bold"),                  # the question text
        ("answer",      "fg:#00d7ff bold"),       # selected answer after enter
        ("pointer",     "fg:#ffd166 bold"),       # the arrow ❯
        ("highlighted", "fg:#ffd166 bold"),       # currently-highlighted row
        ("selected",    "fg:#00d7ff"),            # previously-selected (multiselect)
        ("separator",   "fg:#5c6370"),
        ("instruction", "fg:#5c6370 italic"),
        ("text",        ""),
    ]
)


def _family_emoji(family: str) -> str:
    return {"yolo": "🟢", "rfdetr": "🟣"}.get(family, "⚪")


# ── Menus ───────────────────────────────────────────────────────────────────

def _main_menu() -> str | None:
    return questionary.select(
        "What would you like to do?",
        choices=[
            Choice("🗜️   Compress a model",        "compress"),
            Choice("📊   Benchmark variants",      "benchmark"),
            Choice("🧪   Validate accuracy",       "validate"),
            questionary.Separator("─" * 30),
            Choice("🔄   Refresh model list",      "refresh"),
            Choice("🚪   Exit",                     "exit"),
        ],
        default="compress",
        style=_Q_STYLE,
        qmark="❯",
    ).ask()


def _pick_group(groups: list[ModelGroup]) -> ModelGroup | None:
    if not groups:
        console.print(
            "\n[yellow]⚠  No ONNX files found under[/yellow] "
            f"[cyan]{PROJECT_ROOT}[/cyan].\n"
        )
        return None

    choices: list = []
    for g in groups:
        variants = ", ".join(sorted({f.variant for f in g.files}))
        title = (
            f"{_family_emoji(g.family)}  "
            f"{g.name:<22}  "
            f"{g.total_size_mb:>7.1f} MB   "
            f"{len(g.files)} file(s) [{variants}]"
        )
        choices.append(Choice(title=title, value=g))

    choices.append(questionary.Separator("─" * 30))
    choices.append(Choice("← Back", value=None))

    return questionary.select(
        "Select a model:",
        choices=choices,
        style=_Q_STYLE,
        qmark="❯",
        use_shortcuts=True,
    ).ask()


def _pick_file(group: ModelGroup) -> ONNXFile | None:
    if len(group.files) == 1:
        return group.files[0]

    choices: list = []
    for f in group.files:
        title = f"{f.variant:<15} {f.size_mb:>7.2f} MB   {f.path.name}"
        choices.append(Choice(title=title, value=f))
    choices.append(questionary.Separator("─" * 30))
    choices.append(Choice("← Back", value=None))

    return questionary.select(
        f"Select an ONNX file from {group.name}:",
        choices=choices,
        style=_Q_STYLE,
        qmark="❯",
    ).ask()


def _pick_stage(props: ONNXProperties) -> str | None:
    """Stage-selection menu, data-driven from the STAGES registry.

    Each registered stage gets one row. ``can_run(props)`` decides
    whether the row is enabled or disabled-with-reason. New stages
    added to ``Compression/stages/`` show up here automatically.
    """
    choices: list = []

    for stage_cls in STAGES.values():
        stage = stage_cls()
        ok, reason = stage.can_run(props)
        title = f"{stage.emoji}   {stage.description}"
        if ok:
            choices.append(Choice(title=title, value=stage.name))
        else:
            choices.append(Choice(title=title, value=stage.name, disabled=reason))

    # Placeholder — TensorRT lives outside the Stage registry for now
    # (it's a CUDA-engine path, not an ONNX-to-ONNX transform).
    choices.append(Choice(
        "⚡   TensorRT compile  (deferred — wires existing src/inference/export_engine.py)",
        value="tensorrt",
        disabled="not yet wired into the Compression tool",
    ))

    choices.append(questionary.Separator("─" * 30))
    choices.append(Choice("← Back", value=None))

    return questionary.select(
        "Select a compression stage:",
        choices=choices,
        style=_Q_STYLE,
        qmark="❯",
    ).ask()


# ── Actions ─────────────────────────────────────────────────────────────────

def _handle_compress(groups: list[ModelGroup]) -> list[ModelGroup]:
    """Three-level back-navigable state machine.

        Pick Group  ──▶ (None = back to main)
           └── Pick File  ──▶ (None = back to group picker)
                   └── Properties + Stage picker
                           └── (None = back to file picker)
                               └── (stage picked = run, then stay)

    After every successful stage we re-discover ``groups`` so the new
    output file shows up in the table without forcing a manual refresh.
    The updated list is returned so the caller (``main``) can keep its
    own copy in sync.
    """
    while True:                                           # level 1 — group
        group = _pick_group(groups)
        if group is None:
            return groups                                  # ← back to main menu

        while True:                                       # level 2 — file
            onnx_file = _pick_file(group)
            if onnx_file is None:
                break                                      # ← back to group picker

            # Inspect the source ONCE on file selection — properties don't
            # change while we're inside this loop. Re-printing every iteration
            # was noisy; the post-stage summary tells the operator what
            # actually changed.
            print_group_detail(group)
            try:
                src_props = inspect_onnx(onnx_file.path)
            except Exception as e:
                console.print(
                    f"\n[red]❌  Failed to inspect[/red] "
                    f"[cyan]{onnx_file.rel}[/cyan]: [dim]{e}[/dim]\n"
                )
                continue                                   # back to file picker
            print_properties(src_props)

            while True:                                   # level 3 — stage picker
                stage = _pick_stage(src_props)
                if stage is None:
                    break                                  # ← back to file picker

                out_path = _execute_stage(stage, onnx_file, src_props)

                if out_path is not None:
                    # 1. Refresh the discovered groups so the file picker
                    #    sees the new variant on the next "← Back" trip.
                    groups = discover_onnx(PROJECT_ROOT)
                    refreshed = next(
                        (g for g in groups if g.source == group.source), None
                    )
                    if refreshed is not None:
                        group = refreshed

                    # 2. Show the OUTPUT properties — what the new artefact
                    #    looks like vs the source (size delta, weight dtype
                    #    change, op count delta, new op types introduced).
                    try:
                        out_props = inspect_onnx(out_path)
                        print_output_summary(out_props, src_props)
                    except Exception as e:
                        console.print(
                            f"[dim]Could not inspect output: {e}[/dim]"
                        )

                    # 3. Show the refreshed variant list inline so the
                    #    operator sees the new file slot in.
                    console.print("\n[dim]🔄  Refreshed variant list:[/dim]")
                    print_group_detail(group)

                # Stay at level 3 so operator can chain stages on the same
                # source (e.g. FP16 then INT8 then simplify). Ctrl+C or
                # picking "← Back" exits cleanly.


def _stage_run_info(stage) -> str | None:
    """Render a one-liner describing anything noteworthy about the run.

    Hook for future stages that want to surface their own context
    (calibration source, GPU target, etc.). Returns ``None`` for stages
    that don't need pre-run info.
    """
    if getattr(stage, "name", "") == "int8":
        return "dynamic mode — no calibration data needed"
    return None


def _execute_stage(stage_key: str, onnx_file: ONNXFile, props: ONNXProperties) -> Path | None:
    """Run a registered stage on ``onnx_file``.

    Flow: registry lookup → pre-flight ``can_run`` → overwrite confirm →
    run with spinner → report size delta. Any exception from the stage
    is caught and rendered; it never crashes the menu loop.

    Returns the output ``Path`` on success so the caller can refresh
    the discovered model groups. Returns ``None`` if the stage was
    skipped, cancelled, or failed.
    """
    stage_cls = STAGES.get(stage_key)
    if stage_cls is None:
        console.print(
            f"\n[dim italic]🚧  Stage [bold]{stage_key}[/bold] not yet implemented — "
            f"coming soon.[/dim italic]\n"
        )
        return None

    stage = stage_cls()

    # 1. Pre-flight check — skip stages that would be no-ops or impossible.
    ok, reason = stage.can_run(props)
    if not ok:
        console.print(
            f"\n[yellow]⚠  Cannot run {stage.emoji} {stage.name}:[/yellow] "
            f"[dim]{reason}[/dim]\n"
        )
        return None

    src = onnx_file.path
    out = stage.output_path(src)

    # 2. Overwrite check — explicit confirmation keeps the tool idempotent.
    if out.exists():
        overwrite = questionary.confirm(
            f"{out.name} already exists. Overwrite?",
            default=False,
            style=_Q_STYLE,
            qmark="❯",
        ).ask()
        if not overwrite:
            console.print("[dim]   Cancelled.[/dim]\n")
            return None

    # 3. Run with a spinner. Any exception bubbles up and is rendered.
    console.print(
        f"\n{stage.emoji}  [bold]{stage.name}[/bold]  "
        f"[dim]{src.name}  →  {out.name}[/dim]"
    )

    # Extra context for stages that rely on external data (int8 calibration).
    extra = _stage_run_info(stage)
    if extra:
        console.print(f"   [dim]{extra}[/dim]")

    try:
        with console.status(
            f"[dim]Running {stage.name}…[/dim]",
            spinner="dots",
            spinner_style="cyan",
        ):
            result = stage.run(src)
    except Exception as e:  # noqa: BLE001 — render every failure, never crash the loop
        console.print(f"[red]❌  {stage.name} failed: {e}[/red]\n")
        return None

    # 4. Report size delta so operators can see the win at a glance.
    new_mb = result.stat().st_size / (1024 * 1024)
    old_mb = src.stat().st_size / (1024 * 1024)
    delta_pct = (new_mb / old_mb - 1) * 100 if old_mb else 0.0
    verb = "grew" if delta_pct > 0 else "shrank"
    colour = "yellow" if delta_pct > 0 else "green"
    try:
        rel = result.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = result

    console.print(
        f"✅  [green]Wrote[/green] [cyan]{rel}[/cyan] "
        f"[dim]({old_mb:.2f} → {new_mb:.2f} MB, "
        f"[{colour}]{verb} {abs(delta_pct):.1f}%[/{colour}])[/dim]\n"
    )
    return result


def _pick_benchmark_iterations() -> int | None:
    """Let the operator choose a quick/thorough benchmark pass."""
    return questionary.select(
        "How many iterations per variant?",
        choices=[
            Choice("🚀  Quick    — 20  iters × 1 frame",   20),
            Choice("⚖️   Standard — 50  iters × 1 frame",   50),
            Choice("🔬  Thorough — 200 iters × 1 frame",   200),
            questionary.Separator("─" * 30),
            Choice("← Back", value=None),
        ],
        default=50,
        style=_Q_STYLE,
        qmark="❯",
    ).ask()


def _handle_benchmark(groups: list[ModelGroup]) -> list[ModelGroup]:
    """Benchmark every ONNX variant in a chosen model group.

    Back-nav model mirrors _handle_compress:
        Pick Group → Pick Iterations → Run → loop back to iterations
        (so operator can re-run quick/thorough without re-picking the model)
    """
    while True:                                       # level 1 — group
        group = _pick_group(groups)
        if group is None:
            return groups                              # ← back to main menu

        while True:                                   # level 2 — iteration count
            iterations = _pick_benchmark_iterations()
            if iterations is None:
                break                                  # ← back to group picker

            _run_benchmark(group, iterations)
            # stay on iteration picker so operator can re-run quickly


def _run_benchmark(group: ModelGroup, iterations: int) -> None:
    """Execute the benchmark and render the results table."""
    console.print(
        f"\n📊  Benchmarking [bold]{group.name}[/bold] "
        f"[dim]({len(group.files)} variant(s), {iterations} iters each)…[/dim]\n"
    )

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running", total=len(group.files))

        def _tick(done: int, total: int, name: str) -> None:
            progress.update(task, completed=done, description=f"Running [cyan]{name}[/cyan]")

        try:
            results, sample_path = run_group_benchmark(
                group,
                iterations=iterations,
                on_progress=_tick,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]❌  Benchmark crashed:[/red] [dim]{e}[/dim]\n")
            return

    print_benchmark_results(
        group_name=group.name,
        results=results,
        sample_path=sample_path,
        iterations=iterations,
    )


def _pick_baseline(group: ModelGroup) -> ONNXFile | None:
    """Pick which file in the group counts as the ground truth."""
    # Default: the 'base' variant if present.
    default = next((f for f in group.files if f.variant == "base"), None)

    choices: list = []
    for f in group.files:
        tag = "  ← default" if f is default else ""
        choices.append(Choice(
            title=f"{f.variant:<15} {f.size_mb:>7.2f} MB   {f.path.name}{tag}",
            value=f,
        ))
    choices.append(questionary.Separator("─" * 30))
    choices.append(Choice("← Back", value=None))

    return questionary.select(
        "Choose the BASELINE (ground truth):",
        choices=choices,
        default=default,
        style=_Q_STYLE,
        qmark="❯",
    ).ask()


def _pick_candidate(group: ModelGroup, baseline: ONNXFile) -> ONNXFile | None:
    """Pick which variant to test against the baseline."""
    others = [f for f in group.files if f is not baseline]
    if not others:
        console.print(
            "[yellow]⚠  Only one file in this group — nothing to validate against.[/yellow]\n"
        )
        return None

    choices: list = []
    for f in others:
        choices.append(Choice(
            title=f"{f.variant:<15} {f.size_mb:>7.2f} MB   {f.path.name}",
            value=f,
        ))
    choices.append(questionary.Separator("─" * 30))
    choices.append(Choice("← Back", value=None))

    return questionary.select(
        "Choose the CANDIDATE to validate:",
        choices=choices,
        style=_Q_STYLE,
        qmark="❯",
    ).ask()


def _pick_validation_frames() -> int | None:
    return questionary.select(
        "How many frames should we diff?",
        choices=[
            Choice("🚀  Quick    — 10  frames",    10),
            Choice("⚖️   Standard — 50  frames",   50),
            Choice("🔬  Thorough — 200 frames",  200),
            questionary.Separator("─" * 30),
            Choice("← Back", value=None),
        ],
        default=50,
        style=_Q_STYLE,
        qmark="❯",
    ).ask()


def _handle_validate(groups: list[ModelGroup]) -> list[ModelGroup]:
    """Four-level back-navigable validation flow.

        Pick Group → Pick Baseline → Pick Candidate → Pick Frames → Run
    """
    while True:                                       # level 1 — group
        group = _pick_group(groups)
        if group is None:
            return groups                              # ← back to main menu

        if len(group.files) < 2:
            console.print(
                f"\n[yellow]⚠  [bold]{group.name}[/bold] has only "
                f"{len(group.files)} file(s) — need at least 2 to validate.[/yellow]\n"
            )
            continue

        while True:                                   # level 2 — baseline
            baseline = _pick_baseline(group)
            if baseline is None:
                break                                  # ← back to group picker

            while True:                               # level 3 — candidate
                candidate = _pick_candidate(group, baseline)
                if candidate is None:
                    break                              # ← back to baseline picker

                while True:                           # level 4 — frame count
                    n_frames = _pick_validation_frames()
                    if n_frames is None:
                        break                          # ← back to candidate picker

                    _run_validation(baseline, candidate, n_frames)
                    # stay on frame picker so operator can re-run fast


def _run_validation(baseline: ONNXFile, candidate: ONNXFile, n_frames: int) -> None:
    """Execute the validation pair diff and render the verdict panel."""
    console.print(
        f"\n🧪  Validating [cyan]{candidate.variant}[/cyan] "
        f"against baseline [cyan]{baseline.variant}[/cyan]…"
    )

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]Diffing detections"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Validating", total=n_frames)

        def _tick(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        try:
            result = validate_pair(
                baseline_file=baseline,
                candidate_file=candidate,
                n_frames=n_frames,
                on_progress=_tick,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]❌  Validation failed:[/red] [dim]{e}[/dim]\n")
            return

    print_validation_result(result)


# ── Main loop ───────────────────────────────────────────────────────────────

def _show_models(groups: list[ModelGroup]) -> None:
    if not groups:
        console.print(
            "\n[yellow]⚠  No ONNX files found under[/yellow] "
            f"[cyan]{PROJECT_ROOT}[/cyan]."
        )
        console.print(
            "[dim]   Drop a .onnx somewhere under models/ or runs/ and refresh.[/dim]\n"
        )
        return
    console.print(build_models_table(groups))


def main() -> None:
    print_banner()
    print_capabilities()

    groups = discover_onnx(PROJECT_ROOT)
    _show_models(groups)

    while True:
        choice = _main_menu()
        if choice is None or choice == "exit":
            break
        if choice == "refresh":
            groups = discover_onnx(PROJECT_ROOT)
            _show_models(groups)
            continue
        # Handlers may re-discover internally (compress writes new files);
        # they return the updated list so main() picks up the refresh too.
        if choice == "compress":
            groups = _handle_compress(groups)
        elif choice == "benchmark":
            groups = _handle_benchmark(groups)
        elif choice == "validate":
            groups = _handle_validate(groups)

    console.print("\n[bold cyan]👋  Goodbye[/bold cyan]\n")


if __name__ == "__main__":
    main()
