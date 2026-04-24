# Unified Vision Engine

The `VisionEngine` is the "brain" of the IsiDetector platform. It unifies the detection logic so that your Command Line scripts and your Web Application behave identically.

---

## Architecture

:material-file-code: **Source**: `isidet/src/shared/vision_engine.py`

The engine encapsulates the entire computer vision pipeline into a single, stateful class:

```mermaid
graph TD
    Frame[Raw Video Frame] --> Engine[VisionEngine]
    Engine --> Inf[Inferencer]
    Inf --> Track[ByteTrack]
    Track --> Line[LineZone Counter]
    Line --> Log[EventLogger]
    Log --> Output[Annotated Frame + Events]
```

### Key Functionalities

#### 1. Tracking Continuity
By wrapping the `sv.ByteTrack` module, the engine assigns persistent IDs to every parcel (e.g., `#104`). This ensures that even if a parcel is briefly obscured, the system remembers it until it crosses the counting line.

#### 2. Trigger Management
The `process_frame()` method handles the precise logic of "Line Crossing".
- It detects when an object's **Bottom Center** point crosses the designated vertical line.
- It ensures an object is **only counted once** per session, even if it hovers over the line.
- The internal `counted_ids` set is automatically pruned when it exceeds 50,000 entries — the lower half (oldest IDs) is discarded. Since ByteTrack IDs are monotonically increasing and never reused, this is safe for sessions lasting many hours.

#### 3. Real-Time Visualization
The engine generates a fully annotated frame including:
- **Segmentation Masks**: Transparent color overlays on the parcels.
- **Tracing**: Visual lines showing the path the parcel has taken.
- **Labels**: High-contrast text overlays with Object IDs and Class Names.

#### 4. Persistent Hot-Swap
The `swap_inferencer(new_inferencer)` method replaces the model **in place** without tearing down session state. The tracker, `counted_ids` set, line zone, and daily CSV logger all survive — counts keep accumulating and tracker IDs stay stable across the swap. Only the inferencer reference and the palette-dependent annotators (`mask_annotator`, `box_annotator`, `label_annotator`) are rebuilt so that class-ID colours reflect the new model's convention.

This lets operators swap between YOLO (`.pt`, `.onnx`) and RF-DETR (`.pth`, `.onnx`) mid-shift with no loss of session data and no double-counts for objects mid-crossing during the transition.

See [Platform Functionalities → Persistent Model Hot-Swap](../web-app/functionalities.md#persistent-model-hot-swap) for the end-to-end operator-facing behaviour.

---

## Implementation Example

```python
from src.shared.vision_engine import VisionEngine
from src.inference.onnx_inferencer import ONNXInferencer

# 1. Initialize
engine = VisionEngine(
    inferencer=ONNXInferencer("best.onnx"),
    config=my_config_dict
)

# 2. Process
annotated, detections, event = engine.process_frame(frame, counts)

# 3. Handle Events
if event:
    print(f"Parcel {event['id']} ({event['class']}) crossed the line!")

# 4. Shutdown — flushes the final CSV snapshot
engine.cleanup(counts)
```

---

## API Reference

::: src.shared.vision_engine.VisionEngine
