# IsiDetector — Deck v4 (client version)

Cover + 4 slides. Dark navy `#0B0F19` · cards `#121A2C` / border `#26334E` · pill badges · Segoe UI ·
one accent per slide · top accent bar · `0X / 04` counter. Fill ⟨placeholders⟩ before presenting.
Rule: headline = one sentence · card = ≤ 6 words.

---

## COVER (keep v3 cover)
**IsiDetector** — Automated Parcel Sorting by Camera
01 The Stake · 02 The Promise · 03 The Proof · 04 The Confidence

---

## 01 · THE STAKE — orange
**Your line sorts well — the last few percent hide in the labels.**

- `TODAY` — Sensor + barcode. A proven chain.
- `THE PHYSICS` — Folded, damaged, face-down labels → no-read.
- `THE OPPORTUNITY` — Recover ⟨2⟩ % no-reads = ⟨60⟩ parcels/h untouched.

Right: 2 site photos — CARTON · POLYBAG.

*Say: credit their chain first. No-reads are physics, every hub has them. Ask their real
no-read rate, use their number. Pitch = recovery, not replacement.*

---

## 02 · THE PROMISE — blue
**A second pair of eyes — it recognises the parcel, no label needed.**

Diagram: `[ Your camera ] → [ IsiDetector · your PC ] → [ Your sorter ]`
caption: *network message · or electrical pulse*

- `COMPLEMENT` — Your chain stays in charge.
- `NO NEW HARDWARE` — Your PC · any IP camera.
- `YOUR MACHINE'S LANGUAGE` — UDP or relay pulse.

*Say: a product you install, not a research project. No operator — starts itself, survives reboots.*

---

## 03 · THE PROOF — green  ← demo
**It is already running on your line.**

- **> 9.5 / 10** — parcels correctly identified
- **× 1** — counted once, never twice
- **~ 15 ms** — decision time; the camera is the limit

Bottom: `▶ LIVE DEMO` — video placeholder.

*Say: "Rather than a slide — let me show you." → video, UDP window, UDP + WIRE LEDs.*

---

## 04 · THE CONFIDENCE — indigo
**It tells us what's wrong before you call us.**

- `SELF-DIAGNOSIS` — 7 health lights · root cause in 30 s.
- `YOUR DATA` — 30-day log · one-click export.
- `SAFE BY DEFAULT` — Auto-restart · remote support.

`GROWS WITH YOU`
- `NEW PARCEL TYPES` — Teach it from your images.
- `MULTI-LINE · MULTI-SITE` — Same install everywhere.

Pills: `FR / EN` · `upgrade without stopping` · `glare handling` · `zone control`

*Say: the 6-am story — dashboard shows camera vs PC vs software in 30 s, no site visit.
Close on confidence, open the discussion.*

*Q&A ammo (never on slides): two AI engines · live sorter retarget · per-parcel ID dedup ·
network lock-down tool · built-in manual · µs latency histograms.*

---
---

# GENERATION PROMPT

Create a 4-slide client deck (16:9) for "IsiDetector" — industrial computer vision that identifies
parcels (carton vs polybag) on a conveyor and triggers the client's sorter in real time. It
complements the client's existing sensor + barcode system by recovering no-reads.

AUDIENCE: client management + automation engineer. NEVER write: YOLO, ByteTrack, mAP, ONNX, Docker,
inference, model. Minimal text: headline = one sentence; card = one label + ≤ 6 words. No paragraphs.

DESIGN: background #0B0F19; cards #121A2C, 1px #26334E border, 8px radius; Segoe UI; titles bold
white 30pt, subs #CBD5E1 13pt; pill badges #142644 with bold 9pt accent text; 4px top accent bar;
"PART 0X · NAME" pill left, "0X / 04" right (#94A3B8). Accents: S1 #FB923C, S2 #38BDF8, S3 #34D399,
S4 #6366F1.

SLIDE 1 · THE STAKE (#FB923C)
H: "Your line sorts well — the last few percent hide in the labels."
Cards: [TODAY] "Sensor + barcode. A proven chain." · [THE PHYSICS] "Folded, damaged, face-down
labels → no-read." · [THE OPPORTUNITY] "Recover 2 % no-reads = 60 parcels/h untouched."
Right: two photo frames labeled CARTON, POLYBAG.

SLIDE 2 · THE PROMISE (#38BDF8)
H: "A second pair of eyes — it recognises the parcel, no label needed."
Center diagram: [Your camera] → [IsiDetector · your PC] → [Your sorter], caption "network message ·
or electrical pulse".
Cards: [COMPLEMENT] "Your chain stays in charge." · [NO NEW HARDWARE] "Your PC · any IP camera." ·
[YOUR MACHINE'S LANGUAGE] "UDP or relay pulse."

SLIDE 3 · THE PROOF (#34D399)
H: "It is already running on your line."
3 KPI cards (big number + one line): "> 9.5 / 10 — parcels correctly identified" · "× 1 — counted
once, never twice" · "~ 15 ms — decision time; the camera is the limit".
Bottom wide card "▶ LIVE DEMO" with a 16:9 media placeholder.

SLIDE 4 · THE CONFIDENCE (#6366F1)
H: "It tells us what's wrong before you call us."
Row 1: [SELF-DIAGNOSIS] "7 health lights · root cause in 30 s." · [YOUR DATA] "30-day log ·
one-click export." · [SAFE BY DEFAULT] "Auto-restart · remote support."
Row 2 "GROWS WITH YOU": [NEW PARCEL TYPES] "Teach it from your images." · [MULTI-LINE · MULTI-SITE]
"Same install everywhere."
Bottom pills: "FR / EN" · "upgrade without stopping" · "glare handling" · "zone control".
