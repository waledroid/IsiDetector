# compression/ — Isitec Model Compression Tool

Interactive, multistage CLI for shrinking ONNX models.

## Quickstart

```bash
# 1. Install the single external dep
pip install -r compression/requirements.txt

# 2. Launch the tool
python -m compression          # from the project root
# or equivalently:
./compress.sh                   # root-level shell wrapper
```

The banner announces the tool, the capabilities list previews what each
stage will do, and the detected-models table shows every `.onnx` found
under the project root — grouped by training run so `best.onnx` and
`best.sim.onnx` share a row.

## Main menu

- `compress`  — pick a model, then pick which compression stage to run
- `benchmark` — compare size + FPS across every variant of a model
- `validate`  — diff a compressed variant's predictions vs its baseline
- `refresh`   — rescan the project (useful after dropping a new `.onnx`)
- `exit`      — leave the tool

v1 covers the scaffolding + menu + discovery. Individual stages (FP16,
INT8, TensorRT) arrive in v2.

## Directory layout

```
compression/
├── __init__.py           # __version__, __title__
├── __main__.py           # enables `python -m compression`
├── cli.py                # entry point + menu loop
├── discovery.py          # find + group ONNX files
├── ui.py                 # banner, capabilities, tables
├── stages/
│   ├── __init__.py       # stage registry (STAGES dict + @register)
│   └── base.py           # Stage ABC — contract for v2+ stages
├── requirements.txt      # rich
└── README.md             # this file
```

## Adding a new compression stage (v2 preview)

```python
# compression/stages/fp16.py
from pathlib import Path
from .base import Stage
from . import register
from ..discovery import ONNXFile

@register
class FP16Stage(Stage):
    name = "fp16"
    emoji = "🪶"
    description = "Convert FP32 weights to FP16 — halves size, ~no accuracy loss"

    def can_run(self, onnx_file: ONNXFile) -> tuple[bool, str]:
        if onnx_file.variant == "fp16":
            return False, "This file is already FP16"
        return True, ""

    def run(self, onnx_file: ONNXFile, out_dir: Path) -> ONNXFile:
        # ...implementation using onnxconverter_common.float16...
        ...
```

The `@register` decorator adds the class to the global `STAGES` dict. The
CLI's `_handle_compress` will iterate `STAGES.values()` to build the
stage-selection menu.

## Where compressed outputs land

Alongside the source ONNX. A model at
`isidet/models/yolo/<run>/weights/best.onnx` will produce sibling files
like `best.fp16.onnx`, `best.int8.onnx`, `best.engine` in the same
directory. This matches the existing `isidet/src/inference/export_engine.py`
convention and means the web app's auto-discovery picks them up without
any config changes.

## What's next

- v2: FP16 stage, INT8 quantisation with calibration, TensorRT engine
  compilation via `trtexec`
- v3: Benchmark harness + accuracy validation
- v4: Non-ONNX support (`.pt`, `.pth`) via an export-then-compress path
