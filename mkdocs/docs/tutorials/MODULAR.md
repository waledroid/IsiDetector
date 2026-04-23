# Modular Plug-In Architecture in Python — Registry + Abstract Base Class

A tutorial on the two Python ideas that let IsiDetector's orchestration code (`isidet/scripts/run_train.py`) stay ~100 lines while supporting any number of model families (YOLO, RF-DETR, and whatever comes next). Read this after you've found yourself reaching for `if model_type == "yolo": ... elif model_type == "rfdetr": ...` and wishing it away.

Two patterns, one goal: **write the orchestrator once, plug new models in without touching it.**

- **Abstract Base Class (ABC)** — enforces a shared contract so the orchestrator can call `.train()` on any model and know it will work.
- **Registry** — maps a string name (like `"rfdetr"`) to a class, so the orchestrator can pick the right one *at runtime* from a config file instead of hard-coding imports.

All code citations in this document reference files in this repo. Follow them — the real thing is ~100 lines and will outlast any toy example.

---

## 1. The Problem, Stated Plainly

**Without this pattern**, adding a new trainer means editing a central dispatcher:

```python
# The bad version
def train(config):
    if config['model_type'] == 'yolo':
        from src.training.trainers.yolo import YOLOTrainer
        trainer = YOLOTrainer(config)
    elif config['model_type'] == 'rfdetr':
        from src.training.trainers.rfdetr import RFDETRTrainer
        trainer = RFDETRTrainer(config)
    elif config['model_type'] == 'something_new':
        from src.training.trainers.something_new import SomethingNewTrainer
        trainer = SomethingNewTrainer(config)
    # ... grows forever ...
    trainer.train()
```

Every new model:

1. Requires an edit to the central dispatcher (risk of breaking unrelated trainers).
2. Forces the dispatcher to import every trainer — slow startup, tight coupling.
3. Requires duplicated knowledge: the name `"rfdetr"` lives in the config, in the `elif`, and in the import path.

**With the pattern**, the orchestrator becomes:

```python
trainer_class = TRAINERS.get(config['model_type'])
trainer = trainer_class(config)
trainer.train()
```

Three lines, agnostic to what's registered. See `isidet/scripts/run_train.py:99-100` for the real version.

New models slot in by adding a file and a decorator. No dispatcher edits.

---

## 2. The Two Pieces

### Abstract Base Class — "Here is the contract. Fill in the blanks."

An ABC is a class that declares *what methods subclasses must implement*, without saying how. It's a promise that every concrete child class makes to its callers.

```python
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train(self): ...

    @abstractmethod
    def evaluate(self): ...
```

Any class that inherits from `BaseTrainer` **must** implement `train` and `evaluate` — if it doesn't, Python refuses to instantiate it:

```
TypeError: Can't instantiate abstract class YOLOTrainer with abstract methods train
```

That error fires at `YOLOTrainer(config)` time, not at some later runtime when the missing method would finally get called. The mistake is caught early, and the contract is enforced by the language itself.

### Registry — "Here's a name. Give me back the class."

A registry is a dictionary that maps strings to classes, plus a decorator that populates it.

```python
REGISTRY = {}

def register(name):
    def wrapper(cls):
        REGISTRY[name] = cls
        return cls
    return wrapper
```

Classes put themselves into it when their module is imported:

```python
@register("yolo")
class YOLOTrainer(BaseTrainer): ...
```

Later, anywhere in the code:

```python
Trainer = REGISTRY["yolo"]      # returns the class
t = Trainer(config)              # instantiate it
t.train()
```

The orchestrator knows *nothing* about `YOLOTrainer` the class, the file it lives in, or the import path. It only knows the string `"yolo"` from config.

### Why together

- **ABC without Registry** = clean contract but you still need a big dispatcher to pick the right class.
- **Registry without ABC** = string-to-class lookup that explodes if the class doesn't have the method you expected. Errors happen at the call site, far from the registration.
- **ABC + Registry** = contract enforced at registration *and* clean lookup. New model = new file + one decorator.

---

## 3. ABC Mechanics (Low-Level)

In this codebase: `isidet/src/training/base_trainer.py`.

### The minimum viable ABC

```python
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train(self):
        pass
```

Two key imports: `ABC` (the metaclass machinery) and `abstractmethod` (the marker decorator). Any class that inherits from `ABC` is considered "abstract if it has any `@abstractmethod` that hasn't been overridden".

### What `@abstractmethod` actually does

It sets a flag on the method: `method.__isabstractmethod__ = True`. When Python tries to instantiate a class, it walks the method resolution order, collects all methods still flagged, and if the set is non-empty — raises `TypeError`.

So:

```python
class YOLOTrainer(BaseTrainer):
    pass                        # doesn't override train

YOLOTrainer(cfg)                # ❌ TypeError at construction
```

vs:

```python
class YOLOTrainer(BaseTrainer):
    def train(self):            # overrides the abstract method
        ...

YOLOTrainer(cfg)                # ✅ works
```

The check runs at **instantiation time**, not import time. You can import an incomplete subclass without error — the bomb goes off only when someone actually tries to construct it.

### Shared behaviour lives in the ABC

ABCs can have concrete methods too, and subclasses inherit them normally. `BaseTrainer` uses this heavily:

```python
# isidet/src/training/base_trainer.py:58-84
def __init__(self, config: dict):
    self.config = config
    self.model_name = config.get('model_type', 'unknown_model')
    self.output_dir = Path(config.get('output_dir', 'models')) / self.model_name
    self.output_dir.mkdir(parents=True, exist_ok=True)
    ...
    self.hooks = []
    for h_name in config.get('hooks', []):
        hook_class = HOOKS.get(h_name)
        self.hooks.append(hook_class())
```

Every subclass gets this `__init__` for free. They call `super().__init__(config)` first, then add whatever is model-specific. Output directory management, hook wiring, config parsing — all written once.

The pattern becomes:

- **Abstract methods** (`build_model`, `train`, `evaluate`, `export`, `_inject_framework_hooks`): model-specific, subclasses must implement.
- **Concrete methods** (`_setup_run_dir`, `_flush_memory`, `call_hooks`, `_parse_common_config`): shared plumbing, lives in `BaseTrainer`.

See `base_trainer.py:163-210` for the abstract methods, and `:86-157` for the concrete helpers.

### Signature is part of the contract

`@abstractmethod` only checks *that* the method exists, not that it has the right signature. Python is duck-typed about this. The ABC's docstring specifies the contract:

```python
# base_trainer.py:206-218 (paraphrased)
@abstractmethod
def evaluate(self) -> dict:
    """Return a metrics dictionary with keys:
       - 'mAP50'      — float, detection mAP @ IoU 0.50
       - 'mAP50_95'   — float, detection mAP @ IoU 0.50–0.95
       ...
    """
    pass
```

Subclasses that violate this contract (say, return a list) will pass instantiation but break the downstream code that calls `metrics["mAP50"]`. The language can only enforce so much — the rest is discipline and good docs.

---

## 4. Registry Mechanics (Low-Level)

In this codebase: `isidet/src/shared/registry.py` — 94 lines.

### The core class

```python
# isidet/src/shared/registry.py:7-69 (trimmed)
class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict = dict()

    def register(self, name: str = None):
        def _register(cls):
            key = name if name else cls.__name__
            self._module_dict[key] = cls
            return cls
        return _register

    def get(self, name: str):
        if name not in self._module_dict:
            raise KeyError(f"'{name}' not found in {self._name} registry.")
        return self._module_dict[name]
```

Three things happen:

1. **The `Registry` instance** holds a private dict of string → class.
2. **`register(name)`** returns a decorator factory. Its inner function `_register(cls)` receives the decorated class, adds it to the dict, and returns the class unchanged (decorators must return *something* to replace the original name in the caller's namespace; here we return the same class so nothing else changes).
3. **`get(name)`** looks up the class by key, with a friendly error if it's not there.

### Why `register(name)` is a decorator factory, not a plain decorator

Python decorator syntax:

```python
@TRAINERS.register('yolo')      # note the parens
class YOLOTrainer(BaseTrainer): ...
```

`@TRAINERS.register('yolo')` is evaluated first — it calls `register('yolo')` which returns the inner `_register` function. That returned function is then used as the actual decorator: `_register(YOLOTrainer)` runs, stores the class, returns the class.

If we didn't need to pass `'yolo'`, it could be a simple `@TRAINERS.register` — but we want to decouple the registered name from the class name, so we need the factory.

### Why global singletons

```python
# isidet/src/shared/registry.py:91-93
PREPROCESSORS = Registry('Preprocessors')
TRAINERS = Registry('Trainers')
HOOKS = Registry('Hooks')
```

These are module-level instances. Because Python only imports a module once per process, there's exactly one `TRAINERS` registry in memory for the whole app. Every file that does `from src.shared.registry import TRAINERS` gets the same object. Register from any module, look up from any other — they all see the same dict.

This is the Singleton pattern, implemented cheaply via module-level state. No metaclass gymnastics needed; Python's import system already guarantees it.

### What gets registered, what doesn't

The decorator fires **at import time** of the module it lives in. Until the trainer module is imported, the registry doesn't know about the trainer.

This is why `isidet/scripts/run_train.py:19-23` has:

```python
_TRAINER_MODULES = {
    'yolo':    'src.training.trainers.yolo',
    'yolov26': 'src.training.trainers.yolo',
    'rfdetr':  'src.training.trainers.rfdetr',
}
```

And `:91-95`:

```python
trainer_module = _TRAINER_MODULES.get(model_type)
importlib.import_module(trainer_module)   # triggers @TRAINERS.register('yolo') etc
TrainerClass = TRAINERS.get(model_type)
```

The trainer modules aren't imported until we know we need them. `importlib.import_module` dynamically loads `src.training.trainers.yolo`, which causes `@TRAINERS.register('yolo')` to run, which populates the registry, which is then ready for `.get('yolo')`.

### The discovery vs. invocation tension

You still need *some* file that knows where to find registered classes. There are three ways to solve that:

| Approach | How | Trade-off |
|---|---|---|
| Explicit map (used here) | `_TRAINER_MODULES = {'yolo': 'src.training.trainers.yolo', …}` | Simple, fast startup, one map to maintain |
| Eager import of all | `import src.training.trainers.yolo; import src.training.trainers.rfdetr; …` at top of `run_train.py` | Zero lookup step but imports everything always |
| Directory scan | `for path in Path('src/training/trainers').glob('*.py'): importlib.import_module(...)` | Truly zero-config but surprises lurk if a file fails to import |

IsiDetector uses the explicit map because the orchestrator still gets a list of supported models for free (useful for `--help`, error messages), and startup is fast.

---

## 5. The Full Cycle (High → Low → Ground Level)

Walk the path from a one-line config change to training actually running.

### Level 1: the user's view

Edit `isidet/configs/train.yaml`:

```yaml
model_type: rfdetr
```

Run:

```bash
python isidet/scripts/run_train.py
```

Done.

### Level 2: what the orchestrator does

`isidet/scripts/run_train.py:91-100` roughly:

```python
model_type = config['model_type']                   # 'rfdetr'
trainer_module = _TRAINER_MODULES[model_type]       # 'src.training.trainers.rfdetr'
importlib.import_module(trainer_module)             # fires the decorator
TrainerClass = TRAINERS.get(model_type)             # the class, not an instance
trainer = TrainerClass(config)                      # instantiate
trainer.train()
trainer.evaluate()
trainer.export('onnx')
```

### Level 3: what happens inside `import_module`

Python executes `isidet/src/training/trainers/rfdetr.py` top-to-bottom. At line 25 of that file:

```python
@TRAINERS.register('rfdetr')
class RFDETRTrainer(BaseTrainer):
    ...
```

The decorator runs: `TRAINERS._module_dict['rfdetr'] = RFDETRTrainer`. The class is now findable.

### Level 4: what `TrainerClass(config)` actually does

`RFDETRTrainer.__init__` (not defined — inherits from `BaseTrainer`) runs `BaseTrainer.__init__(config)`. The ABC's `__init__` sets `self.config`, `self.output_dir`, `self.hooks`, etc. Then `RFDETRTrainer.__init__` itself adds `self.model_size`, `self.dataset_path`, `self.history`.

Because `build_model`, `train`, `evaluate`, `export`, `_inject_framework_hooks` are all overridden in `RFDETRTrainer`, the ABC's instantiation check passes.

### Level 5: `trainer.train()`

Python looks up `train` on the instance → finds `RFDETRTrainer.train` → calls it. Polymorphism. The orchestrator wrote `trainer.train()` once and it dispatches to the right implementation based on what class was registered.

---

## 6. In This Codebase — Where Each Thing Lives

| File | What's in it |
|---|---|
| `isidet/src/shared/registry.py` | The `Registry` class and the three singletons (`TRAINERS`, `HOOKS`, `PREPROCESSORS`) |
| `isidet/src/training/base_trainer.py` | `BaseTrainer(ABC)` — the training contract + shared plumbing (output dir, hook wiring, memory flush) |
| `isidet/src/training/trainers/yolo.py` | `@TRAINERS.register('yolo') class YOLOTrainer(BaseTrainer)` |
| `isidet/src/training/trainers/rfdetr.py` | `@TRAINERS.register('rfdetr') class RFDETRTrainer(BaseTrainer)` |
| `isidet/src/training/hooks/*.py` | `@HOOKS.register('IndustrialLogger') class IndustrialLogger:` etc. |
| `isidet/scripts/run_train.py` | Orchestrator. Reads config → imports module → gets class → instantiates → calls contract methods |

The `inference/` side uses ABCs (`isidet/src/inference/base_inferencer.py`) but **not** a registry — instead, file-extension dispatch in `run_live.py`. A deliberate choice: inferencers are selected per-weight-file, not per-config-key, so the extension is a cleaner discriminator than a registered name. Context matters; you don't need to apply both patterns everywhere.

---

## 7. Adding a New Model — The Recipe

Suppose you want to add a `yolov11` trainer. Four steps, none of which touch the orchestrator:

### Step 1: Create the trainer file

`isidet/src/training/trainers/yolov11.py`:

```python
from src.training.base_trainer import BaseTrainer
from src.shared.registry import TRAINERS

@TRAINERS.register('yolov11')
class YOLOv11Trainer(BaseTrainer):

    def build_model(self):
        ...                  # set self.model

    def _inject_framework_hooks(self):
        ...                  # wire native callbacks → self.call_hooks('after_epoch')

    def train(self):
        self._setup_run_dir()
        self._inject_framework_hooks()
        self.call_hooks('before_train')
        ...                  # native training loop
        self.call_hooks('after_train')

    def evaluate(self) -> dict:
        return {'mAP50': ..., 'mAP50_95': ...}

    def export(self, format: str = 'onnx') -> str:
        ...
```

All five abstract methods overridden. Instantiation will succeed.

### Step 2: Register the module path

`isidet/scripts/run_train.py:19-23`:

```python
_TRAINER_MODULES = {
    'yolo':    'src.training.trainers.yolo',
    'yolov26': 'src.training.trainers.yolo',
    'rfdetr':  'src.training.trainers.rfdetr',
    'yolov11': 'src.training.trainers.yolov11',   # <- new line
}
```

### Step 3: Use it

```yaml
# isidet/configs/train.yaml
model_type: yolov11
```

### Step 4: Run

```bash
python isidet/scripts/run_train.py
```

The orchestrator will `importlib.import_module('src.training.trainers.yolov11')`, trigger the decorator, look up `'yolov11'` in the registry, and instantiate `YOLOv11Trainer`. Every hook, every CLI flag, every existing piece of infrastructure just works — because the contract is enforced and the plumbing lives in `BaseTrainer`.

No edits to `BaseTrainer`. No edits to `YOLOTrainer` or `RFDETRTrainer`. No edits to `run_train.py` except the one line of module-path mapping.

---

## 8. Pitfalls & Why They Happen

### "My class isn't in the registry"

Means the module wasn't imported, so the decorator never ran. Two common causes:

- Module path mismatch in `_TRAINER_MODULES`.
- Typo in `@TRAINERS.register('yolo')` vs the key you look up.

Fix: add a debug print at registry's `register()` function (or enable the existing `logger.debug` at `registry.py:67`) — you'll see exactly what got registered and under what name.

### "My subclass can't be instantiated — `TypeError: Can't instantiate abstract class`"

You forgot to override one of the `@abstractmethod` methods. The error names the missing ones:

```
TypeError: Can't instantiate abstract class YOLOv11Trainer with abstract method build_model
```

Fix: implement the listed methods. If you genuinely don't need one (rare), you can override it with `pass` — but think twice, the abstract marker usually exists for a reason.

### "Hooks aren't firing"

`BaseTrainer.__init__` builds `self.hooks` from config at construction time (`base_trainer.py:76-84`). If your config has `hooks: [MyHook]` but `MyHook` isn't registered (the hook module wasn't imported), the hook is silently dropped with an error log. See `isidet/scripts/run_train.py:16`:

```python
import src.training.hooks       # triggers hook decorators
```

That import exists specifically to populate `HOOKS` before any `BaseTrainer` is constructed. If you add a new hook file, the `hooks/__init__.py` needs to import it too (or use the same dynamic-import pattern as trainers).

### "Decorator runs twice / class appears to be registered twice"

A module imported via two different paths (e.g., `src.training.trainers.yolo` and `training.trainers.yolo`) will execute its top-level code twice. The second `@TRAINERS.register('yolo')` overwrites the first — which is exactly why the registry warns on overwrite (`registry.py:64`):

```
⚠️ Overwriting existing module 'yolo' in Trainers registry!
```

Fix: use absolute imports consistently (`from src.training.trainers.yolo import ...`), and don't manipulate `sys.path` in ways that create duplicate import roots.

### "I import the trainer module at the top of `run_train.py` but registration still feels fragile"

It is. Eager import works but couples the orchestrator to every trainer. The lazy `importlib.import_module` approach avoids that, with the trade-off that you need `_TRAINER_MODULES` to map names to paths. Either pattern is valid; just pick one and stay consistent.

---

## 9. When NOT to Use This Pattern

- **One backend forever.** If you have one model and always will, `from models.yolo import YOLOTrainer` directly. Registry adds ceremony without payoff.
- **Classes with wildly different signatures.** ABCs shine when callers can treat subclasses uniformly. If every model needs a completely different constructor or entrypoint, the uniformity is fiction and you'd be better off with a factory function per model.
- **Plugins that need config merging, dependency resolution, or version pins.** You've outgrown a dict. Use a real plugin system (`pluggy`, `stevedore`, `importlib.metadata` entry points).

The IsiDetector pattern is the 80/20: dead-simple, works across 2–20 backends, stays readable. Beyond that scale, reach for heavier tools.

---

## Summary

Two ideas, one outcome:

1. **ABC (`isidet/src/training/base_trainer.py`)** — the *contract*. What every trainer promises to deliver. Enforced at instantiation by Python itself.
2. **Registry (`isidet/src/shared/registry.py`)** — the *lookup*. How config strings turn into classes without a giant dispatcher.

Together they make the orchestrator (`isidet/scripts/run_train.py`) three lines of real work, and they make adding a new model a four-step recipe that doesn't risk breaking anything that already works.

The pattern is not specific to ML. It's the bones of plugin systems, driver frameworks, web routers, and half the Python standard library. Once you see it, you'll see it everywhere.

Three rules of thumb:

- **The ABC names methods subclasses must write.** Everything else (shared `__init__`, helpers, lifecycle hooks) belongs as concrete methods on the base class — DRY it up aggressively.
- **The registry decorator fires at import time.** If your class seems missing, the module wasn't imported.
- **Keep the orchestrator ignorant.** The moment the orchestrator has to know which models exist, you've lost half the benefit. Let the decorator + `importlib.import_module` tell you what's available.

For the real code walk: `isidet/src/shared/registry.py` (read top to bottom, ~94 lines) → `isidet/src/training/base_trainer.py` (skim the abstract methods at the bottom, read the `__init__`) → `isidet/src/training/trainers/yolo.py` to see a concrete implementation → `isidet/scripts/run_train.py:91-100` for the orchestrator call pattern. Hour well spent.
