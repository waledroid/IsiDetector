# 🚀 Start here — Site PC playbook

You're on the **`deploy`** branch — the lean runtime subset of IsiDetector, built for site PCs. No training, no compression, no docs source. Just what it takes to run the inference stack and talk to the sorter.

Two host platforms supported:
- **Ubuntu 22.04 / 24.04** (preferred for GPU + production Linux site PCs)
- **Windows 10 build 19041+ / Windows 11** (CPU-only site PCs; uses Docker Desktop with the WSL2 backend, hidden from the operator)

---

## 📦 One-time install — Linux site PC

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

## 🪟 One-time install — Windows site PC

CPU-only stack on Docker Desktop. The operator doesn't open a Linux terminal, doesn't install Ubuntu, doesn't see WSL.

```text
1. Clone or unzip the deploy branch into C:\logistic (or anywhere convenient).
2. Double-click  Install.bat   (UAC will prompt — accept).
```

`Install.bat` calls `deploy\windows\install.ps1`, which:

- Verifies Windows version (10 build 19041+ / Windows 11).
- Downloads + silent-installs Docker Desktop with the WSL2 backend.
- Writes `%USERPROFILE%\.wslconfig` with site-PC defaults (**4 GB RAM / 2 vCPU / 2 GB swap** on the hidden WSL VM, so it doesn't starve the host).
- Builds the CPU image via `deploy\windows\run_start.ps1`.
- Drops a desktop shortcut to `Start.bat`.

If `Install.bat` ever errors out partway through, you can re-run it — every step is idempotent (skips Docker Desktop install if already present, leaves an existing `.wslconfig` untouched, etc.). To rebuild only the image, run `deploy\windows\run_start.ps1` directly from a non-elevated PowerShell.

To tune the WSL VM (e.g. 8 GB RAM on a 16 GB host), edit `%USERPROFILE%\.wslconfig`, then `wsl --shutdown` from any PowerShell.

---

## 🎬 Daily start

**Linux:**
```bash
cd ~/logistic
./up.sh
```

**Windows:** double-click the **IsiDetector** desktop shortcut (or `Start.bat` at the repo root).

Stack starts, browser opens at **http://localhost:9501** on both platforms.

```bash
docker compose logs -f web    # live logs (works on both — same compose stack)
docker compose ps             # container status
docker compose down           # stop the stack
```

---

## 📹 Camera discovery (once per site)

Plugged a new IP camera into the LAN and don't know its URL? Scan the subnet for RTSP (554) and camera web UIs (80 / 8080):

```bash
sudo apt install nmap -y
# Adjust the subnet to match the site LAN (check with: ip route | grep default)
sudo nmap -p 554,80,8080 --open 192.168.1.0/24
```

Each camera's exact RTSP path comes from its manual. Common shapes:

```
rtsp://<user>:<pass>@<ip>:554/<stream>
rtsp://<ip>:554/user=<user>&password=<pass>&channel=1&stream=0.sdp?
```

**Examples discovered at the current site** (per-site — rediscover on every new install):

```
rtsp://admin:admin@192.168.1.88:554/11
rtsp://192.168.1.108:554/user=admin&password=admin123&channel=1&stream=0.sdp?
```

Paste the URL into **Live Inference → RTSP URL** in the web UI. Change factory `admin/admin` credentials on the camera before leaving site.

---

## 🌐 Network lock-down (once per site, after the LAN is stable)

**Linux only (Ubuntu Desktop with NetworkManager):**

```bash
./net.sh show              # ← what you have now (IP, gateway, DNS, UDP target)
sudo ./net.sh apply        # ← freeze it as static NM config
./net.sh test              # ← 5 reachability checks, incl. live UDP egress probe
./net.sh manual            # ← bilingual (FR/EN) UDP protocol sheet for the automaticien
sudo ./net.sh revert       # ← back to DHCP if you need to
```

`net.sh --help` for the full flag list.

**Windows site PCs:** there's no `net.ps1` yet — set the static IP via the Windows GUI (Settings → Network → Ethernet → IP assignment → Manual). Use the values from your network sheet exactly. The bilingual UDP protocol sheet for the automaticien still lives in the Linux helper; print it from a dev box if you need a paper copy on site.

---

## 🔄 Update

**Linux:**
```bash
cd ~/logistic && git pull && ./up.sh
```

**Windows** (from PowerShell at the repo root):
```powershell
git pull
.\Start.bat
```

`up.sh` / `up.ps1` rebuild Docker layers only if files actually changed, so routine pulls that touch just docs or configs restart in seconds.

---

## 🆘 Troubleshooting

| Symptom | First thing to check |
|---|---|
| `./up.sh` freezes on "Waiting for ONNX preload" | `docker compose logs web \| tail -30` — look for missing weight paths |
| `./net.sh` says "no NetworkManager" | You're on the wrong machine; `net.sh` needs Ubuntu Desktop with NM |
| Web UI loads but the model dropdown is empty | `curl http://localhost:9501/api/models` — and drop weights into `isidet/models/…` |
| Sorter not receiving UDP | `./net.sh test` step 5 (Live UDP egress) tells you exactly where the packet stops |
| **Windows:** `Start.bat` errors with "Docker Desktop not found" | Run `Install.bat` first (it installs Docker Desktop). If already installed, start it manually from the Start menu and wait for the whale tray icon to stop animating. |
| **Windows:** stack runs but the host feels sluggish | `%USERPROFILE%\.wslconfig` is letting WSL eat too much RAM. Lower `memory=` (defaults shipped: 4 GB) and `wsl --shutdown` to apply. |
| **Windows:** `Install.bat` says "log out and back in" | First-time install adds your account to the `docker-users` group; group membership only refreshes at sign-in. Sign out, sign in, run `Start.bat`. |

Full project reference (architecture, training pipeline, compression) lives on the **`main`** branch — see `CLAUDE.md` there.
