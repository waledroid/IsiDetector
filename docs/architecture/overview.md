# System Architecture

This page explains the overall design philosophy and how all the components fit together.

---

## Design Philosophy

IsiDetector follows three core principles:

1. **Config-Driven** — Every tuneable parameter lives in YAML, never hardcoded
2. **Registry Pattern** — Classes register themselves; the entrypoint discovers them at runtime
3. **Strategy Pattern** — One abstract contract, multiple interchangeable implementations

These patterns combine to create a system where **adding a new model architecture requires zero changes to existing code** — you just write a new trainer class and register it.

---

## Full Architecture Diagram

```mermaid
graph TB
    subgraph "🎛️ Configuration Layer"
        YAML["train.yaml<br/>━━━━━━━━━━━━<br/>model_type: rfdetr<br/>optimizer_config: ..."]
        OPT_Y["yolo_optim.yaml"]
        OPT_R["rfdetr_optim.yaml"]
    end

    subgraph "🚀 Orchestration Layer"
        RUN["run_train.py<br/>━━━━━━━━━━━━<br/>1. Load + Merge YAMLs<br/>2. Registry Lookup<br/>3. Execute Pipeline"]
    end

    subgraph "🧩 Registry Layer"
        REG["Registry<br/>━━━━━━━━━━━━<br/>TRAINERS: dict<br/>HOOKS: dict<br/>PREPROCESSORS: dict"]
    end

    subgraph "🏋️ Training Layer"
        BASE["BaseTrainer (ABC)<br/>━━━━━━━━━━━━<br/>build_model()<br/>train()<br/>evaluate()<br/>export()<br/>call_hooks()"]
        YOLO["YOLOTrainer<br/>━━━━━━━━━━━━<br/>Ultralytics engine<br/>Auto data.yaml<br/>Callback bridging"]
        RFDETR["RFDETRTrainer<br/>━━━━━━━━━━━━<br/>Roboflow engine<br/>DINOv2 backbone<br/>Loss curve plotting"]
    end

    subgraph "🪝 Hook Layer"
        HOOK["IndustrialLogger<br/>━━━━━━━━━━━━<br/>before_train<br/>after_epoch<br/>after_train"]
        HOOK_N["Your Custom Hook<br/>━━━━━━━━━━━━<br/>(just register it!)"]
    end

    subgraph "🔍 Inference Layer"
        BINF["BaseInferencer (ABC)"]
        YINF["YOLOInferencer"]
        RINF["RFDETRInferencer"]
    end

    subgraph "🛡️ Preprocessing Layer"
        PRE["SpecularGuard<br/>━━━━━━━━━━━━<br/>CLAHE on L channel<br/>in LAB colour space"]
    end

    YAML -->|"merge"| OPT_Y
    YAML -->|"merge"| OPT_R
    RUN -->|"1. load"| YAML
    RUN -->|"2. lookup"| REG
    REG -->|"returns"| YOLO
    REG -->|"returns"| RFDETR
    BASE -->|"inherits"| YOLO
    BASE -->|"inherits"| RFDETR
    YOLO -->|"broadcasts"| HOOK
    RFDETR -->|"broadcasts"| HOOK
    RFDETR -.->|"broadcasts"| HOOK_N
    BINF -->|"inherits"| YINF
    BINF -->|"inherits"| RINF
    REG -->|"registers"| HOOK
    REG -->|"registers"| PRE
```

---

## The Five Layers

### 1. Configuration Layer

Everything starts in YAML. The master config (`train.yaml`) holds global settings and points to model-specific optimizer configs that get **merged** at runtime.

!!! info "Config Merging"
    The `optimizer_config` field in `train.yaml` points to a secondary YAML file. At startup, `run_train.py` reads both files and calls `config.update(optim_config)`, producing a single flat dictionary that gets passed to the trainer.

[:material-arrow-right: Full Configuration Guide](../config/index.md)

---

### 2. Orchestration Layer

`run_train.py` is the CLI entrypoint that:

1. Parses `--config` and `--resume` arguments
2. Loads and merges YAMLs into one config dict
3. Reads `model_type` from config
4. Calls `TRAINERS.get(model_type)` to get the right class
5. Instantiates and runs the full pipeline: **train → evaluate → export**

It has **no awareness** of what model it's running. It just trusts the registry.

---

### 3. Registry Layer

Three singleton registries act as name → class lookup tables. Classes register themselves with decorators at import time. The entrypoint triggers the imports, then looks up the right class by string name.

[:material-arrow-right: Registry Deep-Dive](registry.md)

---

### 4. Training Layer

The `BaseTrainer` abstract class defines the universal contract. Every trainer must implement four methods: `build_model()`, `train()`, `evaluate()`, and `export()`. The base also handles hook lifecycle management.

[:material-arrow-right: BaseTrainer Deep-Dive](base-trainer.md)

---

### 5. Support Layers

**Hooks** listen to training lifecycle events. **Inferencers** run prediction. **Preprocessors** transform images before they enter the pipeline.

[:material-arrow-right: Hooks](../hooks/index.md) · [:material-arrow-right: Inference](../inference/index.md) · [:material-arrow-right: Preprocessing](../preprocessing/index.md)

---

## Data Flow

Here's exactly what happens when you run `python scripts/run_train.py`:

```mermaid
sequenceDiagram
    participant User
    participant CLI as run_train.py
    participant Config as YAML Files
    participant Reg as TRAINERS Registry
    participant Trainer as Concrete Trainer
    participant Hooks as Hook Chain

    User->>CLI: python scripts/run_train.py
    CLI->>Config: Load train.yaml
    CLI->>Config: Load + merge optimizer YAML
    CLI->>Reg: TRAINERS.get("rfdetr")
    Reg-->>CLI: RFDETRTrainer class

    Note over CLI,Trainer: Instantiation Phase
    CLI->>Trainer: RFDETRTrainer(config)
    Trainer->>Hooks: Instantiate hooks from config['hooks']

    Note over CLI,Hooks: Training Phase
    CLI->>Trainer: trainer.train()
    Trainer->>Hooks: call_hooks('before_train')
    loop Every Epoch
        Trainer->>Trainer: Forward + backward pass
        Trainer->>Hooks: call_hooks('after_epoch')
    end
    Trainer->>Hooks: call_hooks('after_train')

    Note over CLI,Trainer: Post-Training
    CLI->>Trainer: trainer.evaluate()
    CLI->>Trainer: trainer.export('onnx')
```
