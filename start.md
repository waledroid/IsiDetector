# 🚀 Start here — Site PC playbook

You're on the **`deploy`** branch — the lean runtime subset of IsiDetector, built for site PCs. No training, no compression, no docs source. Just what it takes to run the inference stack and talk to the sorter.

---

## 📦 One-time install (per site PC)

```bash
# On a fresh Ubuntu host:
curl -fsSL https://raw.githubusercontent.com/waledroid/IsiDetector/deploy/install.sh | bash
```

Or manually:

```bash
git clone --branch deploy https://github.com/waledroid/IsiDetector.git ~/logistic
cd ~/logistic && ./run_start.sh
```

`run_start.sh` auto-detects CPU vs GPU, installs Docker (+ NVIDIA toolkit on GPU hosts), builds the image, and writes the deployment profile marker. After it finishes, log out and back in so your Docker group membership takes effect.

---

## 🎬 Daily start

```bash
cd ~/logistic
./up.sh
```

Stack starts, browser opens at **http://localhost:9501**.

```bash
docker compose logs -f web    # live logs
docker compose ps             # container status
docker compose down           # stop the stack
```

---

## 🌐 Network lock-down (once per site, after the LAN is stable)

```bash
./net.sh show              # ← what you have now (IP, gateway, DNS, UDP target)
sudo ./net.sh apply        # ← freeze it as static NM config
./net.sh test              # ← 5 reachability checks, incl. live UDP egress probe
./net.sh manual            # ← bilingual (FR/EN) UDP protocol sheet for the automaticien
sudo ./net.sh revert       # ← back to DHCP if you need to
```

`net.sh --help` for the full flag list.

---

## 🔄 Update

```bash
cd ~/logistic && git pull && ./up.sh
```

`./up.sh` rebuilds layers only if files actually changed, so routine pulls that touch just docs or configs restart in seconds.

---

## 🆘 Troubleshooting

| Symptom | First thing to check |
|---|---|
| `./up.sh` freezes on "Waiting for ONNX preload" | `docker compose logs web \| tail -30` — look for missing weight paths |
| `./net.sh` says "no NetworkManager" | You're on the wrong machine; `net.sh` needs Ubuntu Desktop with NM |
| Web UI loads but the model dropdown is empty | `curl http://localhost:9501/api/models` — and drop weights into `isidet/models/…` |
| Sorter not receiving UDP | `./net.sh test` step 5 (Live UDP egress) tells you exactly where the packet stops |

Full project reference (architecture, training pipeline, compression) lives on the **`main`** branch — see `CLAUDE.md` there.
