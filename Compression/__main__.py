"""Entry point for ``python -m Compression``.

Two modes:

* No args → launch the existing interactive questionary menu (``cli.main``).
* ``--model PATH --stage NAME`` → non-interactive one-shot run against an
  arbitrary ONNX file, including paths outside the repo tree.

The one-shot mode is what makes the tool scriptable and model-agnostic:
```
./compress.sh --model /abs/path/to/any.onnx --stage fp16
./compress.sh --model models/yolo/2026-04-01/weights/best.onnx --stage int8_qdq
```
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cli import _print_size_delta, main as interactive_main
from .inspect import inspect_onnx
from .stages import STAGES
from .ui import console


def _run_one_shot(model: Path, stage_name: str, output: Path | None) -> int:
    if not model.exists():
        console.print(f"[red]❌  Model not found: {model}[/red]")
        return 2
    if model.suffix.lower() != ".onnx":
        console.print(
            f"[red]❌  Only .onnx inputs are supported (got {model.suffix}). "
            f"Export your .pt/.pth to ONNX first.[/red]"
        )
        return 2

    stage_cls = STAGES.get(stage_name)
    if stage_cls is None:
        console.print(
            f"[red]❌  Unknown stage '{stage_name}'. "
            f"Valid: {', '.join(sorted(STAGES))}[/red]"
        )
        return 2
    stage = stage_cls()

    # Pre-flight: respect the same can_run() gate as the interactive menu.
    props = inspect_onnx(model)
    ok, reason = stage.can_run(props)
    if not ok:
        console.print(
            f"[yellow]⚠  Cannot run {stage.emoji} {stage.name}:[/yellow] "
            f"[dim]{reason}[/dim]"
        )
        return 1

    # Allow --output to override the stage's auto-named default.
    if output is not None:
        stage.output_path = lambda _src, _out=output: _out  # type: ignore[assignment]

    out = stage.output_path(model)
    console.print(
        f"\n{stage.emoji}  [bold]{stage.name}[/bold]  "
        f"[dim]{model.name}  →  {out.name}[/dim]"
    )
    try:
        result = stage.run(model)
    except Exception as e:  # noqa: BLE001 — report, never stack-trace to operators
        console.print(f"[red]❌  {stage.name} failed: {e}[/red]")
        return 1

    _print_size_delta(model, result)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compress",
        description=(
            "Isitec model compression tool. Run with no arguments for the "
            "interactive menu, or pass --model and --stage for a scripted "
            "one-shot conversion against any ONNX file."
        ),
    )
    p.add_argument(
        "--model",
        type=Path,
        help="Path to an ONNX file. Any absolute or relative path is accepted.",
    )
    p.add_argument(
        "--stage",
        choices=sorted(STAGES),
        help="Compression stage to run (one-shot mode).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to an auto-named file next to --model.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # No args → interactive menu (unchanged behaviour).
    if args.model is None and args.stage is None:
        interactive_main()
        return 0

    # Partial args → explicit error, don't silently fall through to the menu.
    if args.model is None or args.stage is None:
        console.print(
            "[red]❌  --model and --stage must be used together for one-shot mode. "
            "Pass no arguments to use the interactive menu.[/red]"
        )
        return 2

    return _run_one_shot(args.model, args.stage, args.output)


if __name__ == "__main__":
    sys.exit(main())
