# 🚀 Start here — Developer setup

Welcome to **IsiDetector** — modular instance segmentation for industrial parcel detection. You're on the **`main`** branch: full source (training, inference, compression, docs, deploy). For the lean runtime-only subset used on site PCs, see the **`deploy`** branch.

---

## ⚡ 1-minute local run (office workstation)

```bash
git clone --branch main https://github.com/waledroid/IsiDetector.git ~/logistic
cd ~/logistic
./run_start.sh     # Docker + (if GPU) NVIDIA toolkit + build image
./up.sh            # start the stack, open Chrome at http://localhost:9501
```

GPU vs CPU is auto-detected. Force a profile for testing:
```bash
./up.sh --force-cpu       # build & run Dockerfile.cpu (no CUDA needed)
./up.sh --force-gpu       # require CUDA, build Dockerfile + rfdetr sidecar
```

---

## 🧠 Training

```bash
# Pick a model in isidet/configs/train.yaml (YOLO or RF-DETR block), then:
python isidet/scripts/run_train.py
```

Datasets live under `isidet/data/`, run outputs land in `isidet/runs/`. Full walkthrough in [`mkdocs/docs/getting-started.md`](mkdocs/docs/getting-started.md).

---

## 🗜️ Compression & format conversion (office GPU workstation only)

```bash
./compress.sh                                      # interactive menu
./compress.sh --model PATH --stage fp16            # one-shot compression
./compress.sh --model foo.pt --convert pt-openvino # pt → onnx → sim → OpenVINO IR
```

Supports `fp16`, `int8`, `int8_qdq`, `sim`, `openvino_fp16` as compression stages, plus 6 format-conversion pipelines. Full reference: [`mkdocs/docs/compression.md`](mkdocs/docs/compression.md).

---

## 📑 Docs

```bash
cd mkdocs && mkdocs serve    # dev server at http://127.0.0.1:8000
cd mkdocs && mkdocs build    # static site → mkdocs/site/
```

---

## 🚢 Deploying to a site PC

Site PCs track the **`deploy`** branch — runtime-only subset, no training / compression / docs. From a fresh Ubuntu site PC:

```bash
curl -fsSL https://raw.githubusercontent.com/waledroid/IsiDetector/deploy/install.sh | bash
```

This clones the `deploy` branch, installs Docker, and walks through the rest. See the `deploy` branch's own `start.md` for the full on-site playbook.

**When you change runtime-relevant files on `main`** (Dockerfiles, compose, `isidet/src/inference`, `webapp/`, `isidet/configs/`, shell wrappers), refresh `deploy` by replaying the deletion set from commit `a6ead19` — see its commit message for the canonical list.

---

## 🌐 Network lock-down (site PCs, bilingual handshake for the automaticien)

On a deployed site PC:
```bash
./net.sh show             # current IP/gateway/DNS, UDP target
sudo ./net.sh apply       # freeze DHCP → static NetworkManager config
./net.sh test             # 5 reachability checks, incl. live UDP egress probe
./net.sh manual           # bilingual (FR/EN) protocol sheet for the automation engineer
```

`net.sh` is meant for bare-metal Ubuntu Desktop with NetworkManager (the site PCs). On a dev machine with no NM it gracefully tells you you're on the wrong host.

---

## 🗺️ Architecture map

| Bucket | Purpose |
|---|---|
| `isidet/` | ML core — `src/`, `scripts/`, `configs/`, `data/`, `models/`, `runs/`, `logs/` |
| `webapp/` | Flask (`isitec_app/`) + FastAPI (`isitec_api/`) peer backends, share `isidet/src/` via PYTHONPATH |
| `compression/` | Interactive compression tool (`python -m compression`) |
| `mkdocs/` | Documentation source + built site |
| `deploy/` | Dockerfiles, `docker-compose*.yml`, `_impl/*.sh` with site-lifecycle scripts |

Full architectural reference: [`CLAUDE.md`](CLAUDE.md) or the rendered docs in `mkdocs/site/`.
