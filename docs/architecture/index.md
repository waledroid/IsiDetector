---
hide:
  - toc
---

# Architecture

Understand how IsiDetector's layered, modular design works — from the Registry pattern to the BaseTrainer contract.

<div class="grid-container" markdown>
<div class="card" markdown>

### 🔍 System Overview

Full architecture diagram, the five-layer design, and end-to-end data flow.

[:material-arrow-right: Read More](overview.md)

</div>
<div class="card" markdown>

### 🧩 Registry Pattern

How the `@TRAINERS.register()` decorator enables zero-modification extensibility.

[:material-arrow-right: Read More](registry.md)

</div>
<div class="card" markdown>

### 📜 BaseTrainer Contract

The abstract foundation that enforces `build_model`, `train`, `evaluate`, and `export`.

[:material-arrow-right: Read More](base-trainer.md)

</div>
</div>
