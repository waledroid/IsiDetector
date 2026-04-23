"""Entry point for ``python -m compression``.

Three modes:

* No args → launch the interactive questionary menu (``cli.main``).
* ``--stage NAME --model PATH`` → one-shot compression on an ONNX or
  (for ``openvino_fp16``) OpenVINO IR input. Accepts any absolute or
  relative path, inside or outside the repo tree.
* ``--convert MODE --model PATH`` → one-shot format conversion:

    ====================  ========================================
    MODE                  What it does
    ====================  ========================================
    pt-onnx               .pt/.pth  →  .onnx
    pt-sim                .pt/.pth  →  .onnx  →  .sim.onnx
    pt-openvino           .pt/.pth  →  .onnx  →  .sim.onnx  →  .xml
    onnx-sim              .onnx     →  .sim.onnx
    onnx-openvino         .onnx     →  .xml + .bin
    openvino-fp16         .xml (FP32)  →  .xml (FP16 weights)
    ====================  ========================================

Examples:

    ./compress.sh --model models/yolo/x/weights/best.pt   --convert pt-openvino
    ./compress.sh --model models/yolo/x/weights/best.onnx --stage   fp16
    ./compress.sh --model models/yolo/x/openvino/model.xml --convert openvino-fp16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cli import _print_size_delta, main as interactive_main
from .convert_ops import (
    onnx_to_openvino,
    onnx_to_sim,
    openvino_fp16,
    pt_full_pipeline,
    pt_to_onnx,
)
from .inspect import inspect_onnx
from .stages import STAGES
from .ui import console

#: ``--stage`` accepts every Stage name + the OpenVINO FP16 compressor,
#: which lives in ``convert_ops`` rather than the Stage registry because
#: it operates on ``.xml`` instead of ``.onnx``.
_STAGE_CHOICES = sorted(set(STAGES) | {"openvino_fp16"})

#: ``--convert`` modes. Each maps to a callable in ``convert_ops``.
_CONVERT_MODES = (
    "pt-onnx",
    "pt-sim",
    "pt-openvino",
    "onnx-sim",
    "onnx-openvino",
    "openvino-fp16",
)


def _run_one_shot_stage(model: Path, stage_name: str, output: Path | None) -> int:
    """Execute a compression stage (or openvino_fp16) on one file."""
    if not model.exists():
        console.print(f"[red]❌  Model not found: {model}[/red]")
        return 2

    # openvino_fp16 takes a .xml input and lives in convert_ops.
    if stage_name == "openvino_fp16":
        if model.suffix.lower() != ".xml":
            console.print(
                f"[red]❌  --stage openvino_fp16 expects a .xml file "
                f"(got {model.suffix}).[/red]"
            )
            return 2
        console.print(
            f"\n🟦  [bold]openvino_fp16[/bold]  [dim]{model.name}  →  "
            f"{(output or (model.parent / (model.stem + '.fp16.xml'))).name}[/dim]"
        )
        try:
            result = openvino_fp16(model, output_path=output)
        except Exception as e:  # noqa: BLE001 — operator-facing message
            console.print(f"[red]❌  openvino_fp16 failed: {e}[/red]")
            return 1
        console.print(f"✅  [green]Wrote[/green] [cyan]{result}[/cyan]\n")
        return 0

    # Regular ONNX compression stages go through the Stage framework.
    if model.suffix.lower() != ".onnx":
        console.print(
            f"[red]❌  --stage {stage_name} expects an .onnx file "
            f"(got {model.suffix}). Export your .pt/.pth to ONNX first "
            f"with --convert pt-onnx.[/red]"
        )
        return 2

    stage_cls = STAGES.get(stage_name)
    if stage_cls is None:
        console.print(
            f"[red]❌  Unknown stage '{stage_name}'. "
            f"Valid: {', '.join(_STAGE_CHOICES)}[/red]"
        )
        return 2
    stage = stage_cls()

    props = inspect_onnx(model)
    ok, reason = stage.can_run(props)
    if not ok:
        console.print(
            f"[yellow]⚠  Cannot run {stage.emoji} {stage.name}:[/yellow] "
            f"[dim]{reason}[/dim]"
        )
        return 1

    if output is not None:
        stage.output_path = lambda _src, _out=output: _out  # type: ignore[assignment]

    out = stage.output_path(model)
    console.print(
        f"\n{stage.emoji}  [bold]{stage.name}[/bold]  "
        f"[dim]{model.name}  →  {out.name}[/dim]"
    )
    try:
        result = stage.run(model)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]❌  {stage.name} failed: {e}[/red]")
        return 1

    _print_size_delta(model, result)
    return 0


def _run_one_shot_convert(mode: str, model: Path, output: Path | None) -> int:
    """Execute a format-conversion pipeline on one file."""
    if not model.exists():
        console.print(f"[red]❌  Model not found: {model}[/red]")
        return 2

    try:
        ext = model.suffix.lower()
        if mode == "pt-onnx":
            if ext not in (".pt", ".pth"):
                return _extension_error(mode, ext, (".pt", ".pth"))
            console.print(f"\n🧩  [bold]{mode}[/bold]  [dim]{model.name}[/dim]")
            result = pt_to_onnx(model)
            console.print(f"✅  [green]Wrote[/green] [cyan]{result}[/cyan]\n")

        elif mode == "pt-sim":
            if ext not in (".pt", ".pth"):
                return _extension_error(mode, ext, (".pt", ".pth"))
            console.print(f"\n🧩  [bold]{mode}[/bold]  [dim]{model.name}[/dim]")
            onnx_path = pt_to_onnx(model)
            result = onnx_to_sim(onnx_path)
            console.print(f"✅  [green]Wrote[/green] [cyan]{result}[/cyan]\n")

        elif mode == "pt-openvino":
            if ext not in (".pt", ".pth"):
                return _extension_error(mode, ext, (".pt", ".pth"))
            console.print(f"\n🧩  [bold]{mode}[/bold]  [dim]{model.name}[/dim]")
            results = pt_full_pipeline(model)
            console.print(
                f"✅  [green]Wrote[/green] [cyan]{results['openvino']}[/cyan]\n"
            )

        elif mode == "onnx-sim":
            if ext != ".onnx":
                return _extension_error(mode, ext, (".onnx",))
            console.print(f"\n🧩  [bold]{mode}[/bold]  [dim]{model.name}[/dim]")
            result = onnx_to_sim(model)
            console.print(f"✅  [green]Wrote[/green] [cyan]{result}[/cyan]\n")

        elif mode == "onnx-openvino":
            if ext != ".onnx":
                return _extension_error(mode, ext, (".onnx",))
            console.print(f"\n🧩  [bold]{mode}[/bold]  [dim]{model.name}[/dim]")
            result = onnx_to_openvino(model)
            console.print(f"✅  [green]Wrote[/green] [cyan]{result}[/cyan]\n")

        elif mode == "openvino-fp16":
            if ext != ".xml":
                return _extension_error(mode, ext, (".xml",))
            console.print(f"\n🟦  [bold]{mode}[/bold]  [dim]{model.name}[/dim]")
            result = openvino_fp16(model, output_path=output)
            console.print(f"✅  [green]Wrote[/green] [cyan]{result}[/cyan]\n")

        else:
            console.print(
                f"[red]❌  Unknown --convert mode '{mode}'. "
                f"Valid: {', '.join(_CONVERT_MODES)}[/red]"
            )
            return 2

    except Exception as e:  # noqa: BLE001
        console.print(f"[red]❌  {mode} failed: {e}[/red]")
        return 1

    return 0


def _extension_error(mode: str, got: str, want: tuple[str, ...]) -> int:
    console.print(
        f"[red]❌  --convert {mode} expects {' / '.join(want)} "
        f"(got {got}).[/red]"
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compress",
        description=(
            "Isitec model compression & conversion tool. "
            "Run with no arguments for the interactive menu, or pass "
            "--model + (--stage | --convert) for a one-shot scripted run."
        ),
    )
    p.add_argument(
        "--model",
        type=Path,
        help="Path to the input model. Any absolute or relative path is accepted.",
    )
    p.add_argument(
        "--stage",
        choices=_STAGE_CHOICES,
        help=(
            "Compression stage (one-shot mode). Supports ONNX stages from "
            "the Stage registry plus 'openvino_fp16' for OpenVINO IR inputs."
        ),
    )
    p.add_argument(
        "--convert",
        choices=_CONVERT_MODES,
        help=(
            "Format conversion pipeline (one-shot mode). "
            "pt-openvino runs the full .pt → .onnx → .sim.onnx → .xml chain."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to an auto-named file next to the source.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # No action args → interactive menu (unchanged behaviour).
    if args.model is None and args.stage is None and args.convert is None:
        interactive_main()
        return 0

    if args.stage is not None and args.convert is not None:
        console.print(
            "[red]❌  --stage and --convert are mutually exclusive. "
            "Pick one.[/red]"
        )
        return 2

    if args.model is None:
        console.print(
            "[red]❌  --model is required when using --stage or --convert.[/red]"
        )
        return 2

    if args.stage is not None:
        return _run_one_shot_stage(args.model, args.stage, args.output)

    if args.convert is not None:
        return _run_one_shot_convert(args.convert, args.model, args.output)

    # Shouldn't reach here — argparse already gated the combinations above.
    console.print(
        "[red]❌  Either --stage or --convert must be specified with --model.[/red]"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
