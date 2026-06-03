# Reference / Developer

Everything in this section is **office / `main`-branch material** — it covers training, compression, architecture internals, and deep-dive tutorials that do not exist on a site PC running the `fps` branch.

!!! warning "Not for site PCs"
    None of the tooling described here is present on the `fps` (or `deploy`) branch.
    If you are on a site PC, close this section and return to the [daily operations guide](../daily-ops/start-stop.md).

---

## What lives here

| Page | What it covers |
|---|---|
| [Architecture overview](../architecture/overview.md) | Config-Driven + Registry + Strategy pattern, five-layer stack, data flow |
| [Trainers](../trainers/index.md) | `BaseTrainer` contract, `YOLOTrainer`, `RFDETRTrainer`, how to add a new model family |
| [Hooks](../hooks/index.md) | Hook lifecycle (`before_train`, `after_epoch`, `after_train`), `IndustrialLogger`, writing custom hooks |
| [Config reference](../config/index.md) | `train.yaml`, optimizer YAMLs, mode inference YAMLs (`cpu.yaml` / `gpu.yaml`) |
| [Compression](../compression.md) | `./compress.sh` interactive menu, ONNX → OpenVINO → TensorRT export pipeline, `--stage` flags |
| [Inference backends](../inference/index.md) | Five backend classes, auto-discovery priority, ONNX session lifecycle, TensorRT engine notes |
| [Modular architecture tutorial](../tutorials/MODULAR.md) | Step-by-step walkthrough: implement a new trainer + hook from scratch |

---

!!! note "Where to start on `main`"
    New to the codebase? Read [Architecture overview](../architecture/overview.md) first, then follow the [Modular architecture tutorial](../tutorials/MODULAR.md).
    For compression and export, see [Compression](../compression.md).
