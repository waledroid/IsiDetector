# 🚀 IsiDetector — Quick Deployment Guide

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

On a fresh Ubuntu host with internet:

```bash
curl -fsSL https://raw.githubusercontent.com/waledroid/IsiDetector/deploy/install.sh | bash
```

What it does, in order:
1. Preflight (sudo, Ubuntu version, no-WSL2 reminder).
2. `apt install` git + curl if missing.
3. `git clone --branch deploy --depth 1` into `~/logistic`.
4. `chmod +x` the runtime scripts.
5. Asks if you want to run `./run_start.sh` now (Y default).

`run_start.sh` then auto-detects GPU vs CPU via `nvidia-smi`, installs Docker (+ the NVIDIA Container Toolkit on GPU hosts), builds the right image, and writes the deployment marker `.deployment.env`. Log out and back in afterwards so your Docker group membership takes effect.

Override the auto-detection if needed:
```bash
./run_start.sh --force-cpu     # build CPU image even with GPU present
./run_start.sh --force-gpu     # build CUDA image on a no-GPU box (rare)
```

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
./net.sh show              # current IP/gateway/DNS, UDP target
sudo ./net.sh apply        # freeze DHCP → static NetworkManager config
./net.sh test              # 5-step reachability check incl. live UDP egress probe
./net.sh manual            # bilingual (FR/EN) UDP protocol sheet for the automaticien
sudo ./net.sh revert       # back to DHCP
```

### 🪟 Windows

No `net.ps1` yet — set the static IP via the GUI (**Settings → Network → Ethernet → IP assignment → Manual**) using the values from your network sheet. Print the bilingual UDP protocol sheet for the automaticien from a Linux dev box (`./net.sh manual`) if you need a paper copy on site.

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
