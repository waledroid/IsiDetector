# 🚀 IsiDetector — `fps` test branch (CPU-tuning experiment)

> ⚠ **You're on the `fps` branch, not `deploy`.** This is a throwaway test branch
> derived from `deploy` with extra Settings knobs to A/B CPU performance on a
> dedicated site PC, plus RTSP-ingest fixes (`CAP_PROP_BUFFERSIZE=1`,
> `rtsp_transport=tcp`, on-connect stream-info logging) and a `📡 Site Camera`
> default-Start button that pulls its URL from Settings → Camera (so operators
> don't type RTSP URLs on the landing page). Compose project name is `fps` (not
> `deploy`) so this clone can run alongside a working `~/logistic/` install
> without colliding on `docker compose ps` / volumes / networks.
>
> **Site-PC test workflow:**
> ```bash
> # 1. Stop the working stack to free port 9501
> cd ~/logistic && docker compose down
>
> # 2. Clone this branch into a separate folder
> git clone --branch fps https://github.com/waledroid/IsiDetector.git ~/fps
> cd ~/fps && ./up.sh --force-cpu
>
> # 3. In the browser: dev-unlock, then:
> #    • Settings → Camera group:        Default RTSP URL (used by Site Camera button)
> #    • Settings → Performance group:
> #         - CPU Threads slider
> #         - Skip mask drawing  (biggest win on busy belts)
> #         - Skip trace lines    (removes motion-trail polylines)
> #    Save. On Live Inference click ▶ Start (📡 Site Camera is the default
> #    source — uses the saved RTSP URL). Watch FPS in the Performance tab;
> #    container logs include a 📹 Stream: WxH @ FPS codec=… line so you can
> #    confirm what the camera is actually sending.
> ```
>
> **Rollback (under 30 s):**
> ```bash
> cd ~/fps && docker compose down
> cd ~/logistic && ./up.sh --force-cpu
> ```
>
> The two new knobs are **not yet in `deploy`**. If they win, the change set
> gets cherry-picked back to `deploy` later. Until then, treat this branch as
> experimental.

---

You're on the **`deploy`** branch — the lean runtime subset shipped to site PCs. No training, no compression, no docs source. Just the inference stack and the sorter UDP feed.

## 🎯 Supported targets

| Platform | Hardware | Image built | RF-DETR sidecar |
|---|---|---|---|
| Ubuntu 22.04 / 24.04 | NVIDIA GPU + CUDA | `Dockerfile`     | ✅ enabled |
| Ubuntu 22.04 / 24.04 | CPU only (OpenVINO) | `Dockerfile.cpu` | ❌ skipped |
| Windows 10 (build 19041+) / 11 | CPU only (OpenVINO) | `Dockerfile.cpu` | ❌ skipped |

Same web stack, same UDP protocol, same Docker images on every box. Only the bootstrap path differs.

---

## 1️⃣ One-time install

### 🐧 Linux (GPU or CPU — auto-detected)

Three commands on a fresh Ubuntu host with internet (assumes `git` is already installed — `sudo apt install -y git` if not):

```bash
git clone --branch deploy https://github.com/waledroid/IsiDetector.git ~/logistic
cd ~/logistic
./run_start.sh --force-cpu      # drop --force-cpu if the host has an NVIDIA GPU
```

Then **log out and back in** (so the `docker` group membership takes effect), and:

```bash
cd ~/logistic && ./up.sh        # opens the browser at http://localhost:9501
```

That's it. `run_start.sh` installs Docker (+ NVIDIA Container Toolkit on GPU hosts), builds the right image, and writes a deployment marker so future `./up.sh` calls pick the same profile. The deploy branch ships a working trained YOLO model under `isidet/runs/segment/models/yolo/yolo26n_320_200/` — no scp / USB transfer needed; the model dropdowns populate immediately.

### 🪟 Windows (CPU only)

1. Clone or unzip the `deploy` branch into `C:\logistic`:
   ```cmd
   git clone --branch deploy https://github.com/waledroid/IsiDetector.git C:\logistic
   ```
2. **Double-click `Install.bat`** at the repo root. Accept the UAC prompt.

The installer is fully silent and does the following:
- Verifies Windows 10 build 19041+ / Windows 11 (anything older fails fast).
- Downloads and silent-installs Docker Desktop with the WSL2 backend (~5–10 min).
- Writes `%USERPROFILE%\.wslconfig` with conservative caps for the hidden WSL2 VM (**4 GB RAM / 2 vCPU / 2 GB swap**) so it can't starve the host.
- Builds the CPU image.
- Drops a desktop shortcut to `Start.bat`.

The operator never opens a Linux terminal. To rebuild only the image later, run `deploy\windows\run_start.ps1` from a PowerShell prompt.

> ⚠ If `Install.bat` ends with "log out and back in", do that — first-time Docker installs put your account in the `docker-users` group, and group membership only refreshes at next sign-in.

To tune the WSL2 VM later (e.g. 8 GB RAM on a 16 GB host), edit `%USERPROFILE%\.wslconfig` then run `wsl --shutdown`.

---

## 2️⃣ Daily start

| Platform | How to start |
|---|---|
| 🐧 Linux | `cd ~/logistic && ./up.sh` |
| 🪟 Windows | Double-click the **IsiDetector** desktop shortcut (or `Start.bat` at the repo root) |

Stack starts and the browser opens at **http://localhost:9501**.

Inspect / stop (works on both platforms):
```bash
docker compose ps              # container status
docker compose logs -f web     # live logs
docker compose down            # stop the stack
```

---

## 🔌 Boot-to-running auto-start (Linux site PC, hands-free kiosk)

For a fully unattended site PC — power on → boot → desktop opens → stack
running → browser fullscreen on the dashboard, no human input, no
internet — wire up **three independent layers**. Run all three on the
site PC once:

```bash
cd ~/fps               # or ~/logistic — wherever the install lives

sudo ./autostart.sh enable-autologin $USER   # Layer 1: OS skips the login screen
sudo ./autostart.sh enable-systemd           # Layer 2: docker compose at boot via systemd
./autostart.sh enable                        # Layer 3: kiosk Chrome opens at login

./autostart.sh status                        # confirm all three are green
```

Each layer is independent and can be reverted with the matching
`disable-*` subcommand. Together they take cold-boot-to-stream-running
from "wait for kiosk + click Start" down to **~30–40 s with zero clicks**.

### What each layer does

- **Layer 1 — `enable-autologin USER`** writes `AutomaticLogin=` for
  GDM3 / LightDM / SDDM. Takes effect on next reboot. Auto-detects the
  display manager. Removes the operator's only manual step (sitting at
  the login screen). Reverse with `disable-autologin`.

- **Layer 2 — `enable-systemd`** installs
  `/etc/systemd/system/isidetector.service` that runs `docker compose
  up -d` from the install dir, ordered after `docker.service`. The stack
  is up before the desktop session even loads. Cuts ~30 s off cold boot
  vs. waiting for the .desktop autostart to fire. Reverse with
  `disable-systemd`.

- **Layer 3 — `enable`** writes
  `~/.config/autostart/isidetector.desktop`, which the desktop session
  runs ~10 s after login. Opens the dashboard in **kiosk Chrome** —
  fullscreen, no address bar, no tabs, operator can't accidentally
  navigate away. Press **Ctrl+Alt+F2** for a TTY if you need to drop
  out. The entry auto-rewrites itself based on whether Layer 2 is
  present: with systemd installed it uses `--open-only` (browser only,
  no compose race); without, it uses full `up.sh --no-build --kiosk
  --force-cpu` (compose + browser).

### Combined with the in-app **Auto-start stream on boot** toggle

Settings → Camera has a checkbox that makes the **stream itself**
auto-resume on container start (replays the last successful Start). Tick
it and click Start once to record the model. From then on, every
container restart auto-resumes the saved camera + last-used model — no
operator click on the dashboard.

The three OS-level layers above + that one in-app toggle = the fully
hands-free path. Power button → operator-ready in under a minute.

### Why `restart: unless-stopped` in `docker-compose.yml` isn't enough

It only resumes containers that were running before shutdown. On a
**first** boot after install, or after a fresh `docker compose down`,
nothing brings them back. Layer 2 catches that case at boot, Layer 3
opens the browser regardless of whether compose was already up.

---

## 🔁 Switching between `~/logistic` (deploy) and `~/fps` (test branch)

Only one stack runs at a time — both bind port 9501. To swap between the
working **deploy** stack and the **fps** test stack, stop one and start the
other.

**One-time setup** (clone the test branch alongside — only needed once):

```bash
git clone --branch fps https://github.com/waledroid/IsiDetector.git ~/fps
```

**Switch from `~/logistic` (deploy) → `~/fps` (test):**
```bash
cd ~/logistic && docker compose down && cd ~/fps && ./up.sh --force-cpu
```

**Switch back from `~/fps` (test) → `~/logistic` (deploy):**
```bash
cd ~/fps && docker compose down && cd ~/logistic && ./up.sh --force-cpu
```

Each clone keeps its own `settings.json` (RTSP URL, line config, CPU
threads, render toggles) so flipping back and forth doesn't lose your
configuration on either side.

To pull the latest changes on either branch before the next switch:
```bash
cd ~/fps && git pull            # or: cd ~/logistic && git pull
```

When you're happy with `fps` and want it to become the canonical install,
the change set gets cherry-picked back into `deploy` and you can
`rm -rf ~/fps` after pulling on `~/logistic`.

---

## 3️⃣ Drop in your trained models

Models live under `isidet/models/` (or `isidet/runs/.../weights/`). The web UI auto-discovers `.pt`, `.onnx`, `.xml`, `.engine`, `.pth` files and groups them in the **Settings → Modes** dropdowns.

From your office workstation:
```bash
scp -r isidet/models/yolo/<date>/openvino \
    user@<site-ip>:~/logistic/isidet/models/yolo/<date>/
```

| Site target | Recommended format |
|---|---|
| Linux + GPU | TensorRT `.engine` (compiled per host) or ONNX `.onnx` |
| Linux + CPU | OpenVINO `.xml` + `.bin` (fastest CPU backend) |
| Windows + CPU | OpenVINO `.xml` + `.bin` |

> ⚠ **RF-DETR + OpenVINO is rejected at load time** — OpenVINO 2026 mistranslates the transformer's Einsum ops. Use `.onnx` or native `.pth` for RF-DETR instead.

---

## 4️⃣ Camera discovery (once per site)

Sweep the LAN for IP cameras:

```bash
sudo apt install nmap -y                                # Linux
sudo nmap -p 554,80,8080 --open 192.168.1.0/24          # adjust subnet
```

Each camera's exact RTSP path comes from its manual. Common shapes:
```
rtsp://<user>:<pass>@<ip>:554/<stream>
rtsp://<ip>:554/user=<user>&password=<pass>&channel=1&stream=0.sdp?
```

Paste the URL into **Live Inference → RTSP URL** in the web UI. Change factory `admin/admin` credentials on the camera before leaving site.

---

## 5️⃣ Network lock-down (once per site)

### 🐧 Linux (Ubuntu Desktop with NetworkManager)

```bash
./net.sh show              # current IP/gateway/DNS, UDP target (offline-tolerant)
sudo ./net.sh setup        # interactive multi-NIC freeze — prompts per NIC for static/DHCP/skip + IP
sudo ./net.sh apply        # legacy single-NIC freeze (works when there's a default gateway)
./net.sh test              # reachability check incl. live UDP egress probe (offline-tolerant)
sudo ./net.sh revert       # back to DHCP
```

`setup` is the right starting point on a typical site PC (two LAN NICs, no internet uplink). It auto-discovers every physical NIC by name (`enp1s0`, `enp2s0`, etc.), shows current state, then prompts you per NIC. Saves the result with `autoconnect=yes` so it survives reboot.

### 🪟 Windows

No `net.ps1` yet — set the static IP via the GUI (**Settings → Network → Ethernet → IP assignment → Manual**) using the values from your network sheet.

---

## 🌐 Remote access — Tailscale + RustDesk (`remote.sh`)

Once-per-site setup so you can reach the kiosk from anywhere — no port-forward, no public IP, no TeamViewer license.

```bash
sudo ./remote.sh setup            # interactive: tailscale via Gmail SSO
                                  # + rustdesk service + permanent password
./remote.sh status                # show tailscale IP + rustdesk ID
./remote.sh test                  # connectivity probes (read-only)
sudo ./remote.sh remove           # uninstall both
```

**What you get:**
- **Tailscale** — site PC joins your private mesh at a `100.x.x.x` IP. SSH and HTTPS from your laptop just work, no DNS or NAT shenanigans. Free for personal/small-team use; sign in with the Gmail account that owns the tailnet.
- **RustDesk** — full GUI of the kiosk Chrome dashboard. The script enables the systemd service so RustDesk runs even when no operator is logged in.

The script prints the Tailscale IP, RustDesk ID, and a generated permanent password at the end of `setup` — write them down. They're also saved to `/var/log/isidetector/remote-state.json` for the next visit. Idempotent; safe to re-run.

**Optional flags:**
- `--ts-key tskey-auth-...` — pre-generated Tailscale auth key (hands-off install, no SSO click).
- `--rd-password XYZ` — set a specific RustDesk password instead of a random one.
- `--rd-server self.example.com` — point RustDesk at a self-hosted relay instead of the public RustDesk service.

**Site PC must be online** (HTTPS to `pkgs.tailscale.com` and `github.com`) when running `setup`. The script bails with a clear message if not.

---

## 6️⃣ Update

| Platform | Update flow |
|---|---|
| 🐧 Linux | `cd ~/logistic && git pull && ./up.sh` |
| 🪟 Windows (PowerShell at repo root) | `git pull; .\Start.bat` |

`up.sh` / `up.ps1` rebuild Docker layers only if files actually changed — pulls that touch only docs or configs restart the stack in seconds.

---

## 🆘 Troubleshooting

| Symptom | First thing to check |
|---|---|
| `./up.sh` freezes on "Waiting for ONNX preload" | `docker compose logs web \| tail -30` — look for missing weight paths in `isidet/models/` |
| Web UI loads but the model dropdown is empty | `curl http://localhost:9501/api/models`; drop weights into `isidet/models/...` |
| Sorter not receiving UDP triggers | `./net.sh test` step 5 (live UDP egress) tells you exactly where the packet stops |
| `./net.sh` says "no NetworkManager" | Wrong machine — `net.sh` needs Ubuntu Desktop with NM, not WSL2 or Server |
| **Windows:** `Start.bat` says "Docker Desktop not found" | Run `Install.bat` first, or start Docker Desktop manually and wait for the whale icon to settle |
| **Windows:** host feels sluggish after install | `%USERPROFILE%\.wslconfig` is letting the WSL2 VM eat too much RAM. Lower `memory=` and run `wsl --shutdown` |
| **Windows:** `Install.bat` ends with "log out and back in" | First-time Docker install adds you to `docker-users`; group membership refreshes only at next sign-in |

---

Full project reference — architecture, training pipeline, compression, ONNX/OpenVINO tutorials — lives on the **`main`** branch. See `CLAUDE.md` there.
