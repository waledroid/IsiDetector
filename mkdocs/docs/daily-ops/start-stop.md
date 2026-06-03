# Starting & Stopping

Run `./up.sh` every time you want to start IsiDetector; `docker compose down` to stop it cleanly.

---

## Daily start

```bash
cd ~/fps
./up.sh
```

`up.sh` at the repo root is a thin wrapper — the real logic lives in `deploy/_impl/up.sh`.

What it does in order:

1. Reads `deploy/.deployment.env` (written once by `run_start.sh`) to pick CPU or GPU compose.
2. Runs `docker compose up -d --build` with the right overlay files.
3. Tails the `web` container log and waits until a readiness marker appears (see [Readiness wait](#readiness-wait) below).
4. Opens Chrome at `http://localhost:9501`.

Once the browser opens, the system is ready. See [Dashboard](dashboard.md) for what to do next.

---

## `./up.sh` flags

| Flag | When to use |
|---|---|
| *(none)* | Normal daily start — reads `.deployment.env`, auto CPU/GPU. |
| `--force-cpu` | Force CPU compose regardless of GPU hardware. |
| `--force-gpu` | Force GPU compose. Fails if the NVIDIA driver is not reachable. |
| `--no-build` | Skip the `--build` step. Use **only** for the boot-time autostart path where the image is already built and no code has changed. Do not use after a `git pull`. |
| `--kiosk` | Open Chrome fullscreen with no browser chrome (used by the autostart desktop entry). |
| `--open-only` | Skip compose entirely; just wait for port 9501 and open the browser. Used by the desktop autostart entry when systemd already owns the compose lifecycle. |

Environment overrides (set before the command):

| Variable | Default | Purpose |
|---|---|---|
| `NO_BROWSER=1` | unset | Start the stack without opening a browser (headless / SSH sessions). |
| `TIMEOUT_SEC=N` | `300` | Max seconds to wait for the readiness marker. |
| `URL=http://…` | `http://localhost:9501` | Override the URL the browser opens. |

!!! warning "Do not use `--no-build` after a `git pull`"
    Python source, templates, and JS are baked into the image at build time — they are not bind-mounted. Running `--no-build` after pulling new code keeps the **old image**, silently ignoring every change. The rebuild is fast and offline-safe (dep layer is cached; only the COPY layer re-runs). No internet required once the image has been built once on this host.

---

## Choosing a web backend

The container ships two interchangeable backends selected at start time via the `WEB_BACKEND` environment variable.

| Value | Backend | Extra endpoints |
|---|---|---|
| `flask` *(default)* | Flask — `webapp/isitec_app/app.py` | `/video_feed` MJPEG |
| `fastapi` | FastAPI — `webapp/isitec_api/app.py` | `/ws/video` (binary JPEG stream) + `/ws/stats` (500 ms JSON tick) + `/video_feed` MJPEG fallback |

FastAPI is recommended for site deployments — the WebSocket video stream has lower latency than MJPEG polling.

**To switch backends, export the variable before starting:**

```bash
WEB_BACKEND=fastapi ./up.sh
```

Or set it permanently in your shell session:

```bash
export WEB_BACKEND=fastapi
./up.sh
```

**Switching backends does not require a rebuild** — `--no-build` is safe here because the source for both backends is already in the image and the entrypoint dispatcher simply selects which `app.py` to exec. The compose overlay files and image are the same in both cases.

!!! note "Aliases"
    `WEB_BACKEND=api` and `WEB_BACKEND=uvicorn` are accepted and both select FastAPI. Anything else falls back to Flask with a warning in the container log.

---

## Readiness wait

After `docker compose up`, `up.sh` tails the `web` container log and waits for a readiness marker before opening the browser:

=== "GPU host"

    Waits for whichever comes first:

    - `ONNX preload (CUDA kernels warm` — fires after ONNX kernel preload completes (~30 s on first boot).
    - `Running on http://` or `Uvicorn running on` or `Application startup complete` — Flask/FastAPI server banner (fires even when no weights are pre-configured).

=== "CPU host"

    Waits for:

    - `Running on http://` (Flask) or `Uvicorn running on` / `Application startup complete` (FastAPI).

    No ONNX preload runs on CPU.

If the marker does not appear within `TIMEOUT_SEC` seconds, `up.sh` opens the browser anyway with a warning. The stack is still running — check logs if the UI is blank.

---

## Stopping the stack

```bash
docker compose down          # stop and remove containers (full stop)
docker compose stop          # stop containers, keep them (faster restart later)
docker compose start         # resume after `docker compose stop`
```

All commands work from the repo root (`~/fps`) because `compose.yaml` at the root includes the `deploy/` compose files.

!!! tip "Ctrl+C in logs does not stop the stack"
    If you are tailing logs (`docker compose logs -f web`), pressing Ctrl+C only exits the log viewer. Containers keep running in the background.

---

## Restarting after a code change or `git pull`

```bash
cd ~/fps
git pull                       # safe — settings.json is skip-worktree protected
docker compose down && ./up.sh # rebuilds changed layers, brings stack back up
```

The rebuild is layer-cached: `requirements-deploy.txt` rarely changes, so only the small code COPY layer re-runs. Typical restart time after a pull: 15–30 s.

---

## Viewing logs

```bash
docker compose logs -f web        # live log (both Flask and FastAPI write here)
docker compose logs -f rfdetr     # rfdetr sidecar (GPU hosts only)
docker compose logs --tail=200 web
```

---

## Kiosk / autostart

If `autostart.sh` was run during site setup, the system boots hands-free:

- **Layer 2 (systemd)** — `docker compose up -d` runs at boot before the desktop session loads.
- **Layer 3 (desktop entry)** — kiosk Chrome opens on `http://localhost:9501` after login. It invokes `up.sh --open-only` (skips compose, just waits for port 9501 and opens the browser) to avoid racing with the systemd unit.

To check autostart status:

```bash
./autostart.sh status
```

For full autostart setup, see [site-operations.md](../site-operations.md).
