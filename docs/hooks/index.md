# Hook System

The hook system provides a lightweight **Observer pattern** for reacting to training lifecycle events without modifying trainer code. Hooks are registered by name and attached via YAML config.

---

## How Hooks Work

```mermaid
graph LR
    subgraph "train.yaml"
        A["hooks:<br/>  - IndustrialLogger<br/>  - YourCustomHook"]
    end
    subgraph "BaseTrainer.__init__()"
        B["HOOKS.get('IndustrialLogger')"]
        C["Instantiate + append"]
    end
    subgraph "Training Loop"
        D["call_hooks('before_train')"]
        E["call_hooks('after_epoch')"]
        F["call_hooks('after_train')"]
    end

    A --> B --> C --> D --> E --> F
```

### Lifecycle Stages

| Stage | When It Fires | Common Use |
|---|---|---|
| `before_train` | Before first epoch | Print headers, start timers, init connections |
| `before_epoch` | Before each epoch | Reset per-epoch counters |
| `after_epoch` | After each epoch | Log metrics, save checkpoints, send alerts |
| `after_train` | After final epoch | Print summary, close connections, cleanup |

### What Hooks Receive

Every hook method receives the **trainer instance** as its argument:

```python
def after_epoch(self, trainer):
    trainer.current_epoch      # Current epoch number
    trainer.current_loss       # Scalar total loss for this epoch
    trainer.loss_components    # {"box": 0.42, "seg": 0.31, "cls": 0.18, "dfl": 0.09}
    trainer.config             # Full merged config dict
    trainer.output_dir         # Path to output directory
    trainer.model_name         # "yolo" or "rfdetr"
```

!!! tip "Hook Isolation"
    If a hook raises an exception, it is caught, logged, and **training continues**. A broken hook cannot crash a training run.

---

## IndustrialLogger — The Built-In Hook

:material-file-code: **Source**: `src/training/hooks/industrial_logger.py`
:material-tag: **Registry Name**: `"IndustrialLogger"`

A formatted, table-style epoch logger designed for industrial CV workflows:

```python
@HOOKS.register('IndustrialLogger')
class IndustrialLogger:

    def before_train(self, trainer):                        # (1)!
        header = (
            f"\n{'Epoch':>8} {'GPU_mem':>10} {'box_loss':>10} "
            f"{'seg_loss':>10} {'cls_loss':>10} {'dfl_loss':>10} "
            f"{'Instances':>10} {'Size':>8}"
        )
        print(header)
        print("-" * len(header))

    def after_epoch(self, trainer):                         # (2)!
        epoch_str = f"{trainer.current_epoch + 1}/{trainer.config.get('epochs', 30)}"
        lc = getattr(trainer, 'loss_components', {})

        def _fmt(key):
            return f"{lc[key]:>10.4f}" if key in lc else f"{'--':>10}"

        row = (
            f"{epoch_str:>8} {gpu_mem:>10} "
            f"{_fmt('box')} {_fmt('seg')} {_fmt('cls')} {_fmt('dfl')} "
            f"{'--':>10} {trainer.config.get('image_size', 640):>8}"
        )
        print(row)

    def after_train(self, trainer):                         # (3)!
        print("-" * 80)
        logger.info(f"✅ Training Complete. Weights at {trainer.output_dir}")
```

1. Prints a formatted table header at the start of training
2. Reads `trainer.loss_components` for individual loss terms — shows `--` if a term isn't available (e.g. RF-DETR doesn't expose `box`/`dfl`)
3. Prints a completion message with the output directory

**Sample output (YOLO):**

```text
   Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances     Size
---------------------------------------------------------------------------------------
   1/200      4.23G     1.2345     0.8712     0.4321     0.1234         --      640
   2/200      4.23G     1.1892     0.8401     0.4102     0.1198         --      640
```

**Sample output (RF-DETR):**

```text
   Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances     Size
---------------------------------------------------------------------------------------
   1/101      8.41G         --         --         --         --         --      448
   2/101      8.42G         --         --         --         --         --      448
```

---

## Creating a Custom Hook

Adding a new hook requires **three steps**:

### Step 1: Write the Hook Class

```python title="src/training/hooks/slack_alert.py"
import logging
from src.shared.registry import HOOKS

logger = logging.getLogger(__name__)

@HOOKS.register('SlackAlert')
class SlackAlert:
    """Sends a Slack message when training finishes."""

    def __init__(self):
        self.webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK"

    def after_train(self, trainer):
        import requests
        message = (
            f"🏁 Training Complete!\n"
            f"Model: {trainer.model_name}\n"
            f"Final Loss: {trainer.current_loss:.4f}\n"
            f"Weights: {trainer.output_dir}"
        )
        requests.post(self.webhook_url, json={"text": message})
        logger.info("📨 Slack notification sent!")
```

### Step 2: Import It

```python title="scripts/run_train.py"
import src.training.hooks.slack_alert  # Add this line
```

### Step 3: Enable It in Config

```yaml title="configs/train.yaml"
hooks:
  - "IndustrialLogger"
  - "SlackAlert"            # Add this line
```

That's it. The hook system handles the rest.

!!! tip "Hooks Are Optional Per-Stage"
    A hook doesn't need to implement every stage. If `SlackAlert` only has `after_train`, it's simply skipped during `before_train` and `after_epoch` broadcasts.

---

## API Reference

::: src.training.hooks.industrial_logger.IndustrialLogger
