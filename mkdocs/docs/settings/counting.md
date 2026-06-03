# Counting Line & Triggers

Set up the line a parcel must cross to be counted and to fire a sort signal, and tune how reliably crossings are caught.

## The counting line

The line is a single straight bar across the belt. A parcel is counted once, the moment its **leading edge** crosses the line.

Set three things in the Tracking Line panel:

| Setting | Options | What it does |
|---|---|---|
| Orientation | `vertical` / `horizontal` | Vertical = bar runs top-to-bottom (belt moves left/right). Horizontal = bar runs left-to-right (belt moves up/down). |
| Position | `0.0`–`1.0` (drag) | Where the bar sits. `0.5` = middle. Drag the bar on the live view to move it. |
| Belt direction | `left_to_right`, `right_to_left`, `top_to_bottom`, `bottom_to_top` | The way parcels travel. |

Defaults: orientation `vertical`, position `0.5`, direction `left_to_right`.

!!! tip
    Put the line where the camera sees parcels clearly and well-separated — not where they overlap or sit half off-screen. Keep position between `0.1` and `0.9`.

### Leading-edge trigger

The line fires on the **front** of the parcel (the side that reaches the line first), not its centre. Belt direction tells the engine which edge is the front:

| Orientation | Belt direction | Edge that triggers |
|---|---|---|
| vertical | left_to_right | right side |
| vertical | right_to_left | left side |
| horizontal | top_to_bottom | bottom side |
| horizontal | bottom_to_top | top side |

Firing on the front gives the sorter the most reaction time and keeps the trigger timing independent of parcel size.

!!! warning
    Set **belt direction** to match the real belt. If it is wrong, the engine watches the wrong edge and crossings fire late or are missed.

## One datagram per crossing

Each counted crossing sends exactly one UDP message to the sorter, even when two parcels cross in the same video frame — both get their own message. Messages carry a gap-free `seq` number so the sorter can confirm none were lost. See [Network / UDP](../network/udp.md) for the payload format and target.

## Dedup (one count per parcel)

Two layers stop the same parcel being counted twice:

- **Track-ID dedup — always on.** Each ByteTrack ID is counted once for its lifetime. A parcel hovering over the line cannot re-fire.
- **Time guard — optional, shipped OFF.** Suppresses any crossing within a short window (default `300 ms`) of the previous one.

The time guard is off by default because it drops **real** parcels that follow each other closely on a fast belt. Track-ID dedup already prevents double-counting in normal use.

!!! note
    Turn the time guard **on** only if you see duplicate counts caused by ID churn — when one parcel briefly loses its track and gets a fresh ID at the line. Otherwise leave it off so you do not lose tightly-spaced parcels.

## Count interpolation (low-FPS recovery)

**Shipped ON.** Standard line-crossing needs to catch the exact frame where the parcel flips from one side of the line to the other. On a slow camera or with a dropped frame, a fast parcel can jump the whole line in one step and be missed.

With interpolation on, the engine remembers each parcel it saw **before** the line and counts it as soon as it appears **after** the line — no matter how many frames were skipped between. It shares the same dedup path, so a parcel caught both ways is still counted once.

!!! tip
    Leave this on. Turn it off only if you are debugging counts and want strict frame-to-frame crossing.

## Tracking (ByteTrack)

Tracking assigns the persistent `#ID` to each parcel. Two settings affect counting:

| Setting | Default | Guidance |
|---|---|---|
| Frame rate | auto | Auto-calibrates to the camera's real FPS so the tracker predicts parcel movement correctly. Set a manual value only if the camera reports a wrong FPS. |
| Track buffer | `60` | How many frames a parcel's ID is kept after it disappears. Raise it if IDs drop during brief occlusions; lower it if old IDs linger and cause mismatches. |

!!! note
    Frame-rate auto-calibration matters on fast belts: it lets the tracker keep a parcel's ID across the line instead of losing it and re-counting.

## Quick reference: when to change what

| Symptom | Try |
|---|---|
| Counts too late / missed at fast speed | Check belt direction; keep interpolation on; let frame rate auto-calibrate. |
| Same parcel counted twice | Track-ID dedup is always on — if it still happens, enable the time guard. |
| Distinct close parcels under-counted | Make sure the time guard is **off**. |
| IDs flicker / drop during occlusion | Raise track buffer. |

To restrict counting to one area of the frame, see [Image & ROI](image-roi.md).
