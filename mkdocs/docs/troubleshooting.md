# Troubleshooting

Symptom-to-fix runbook for site operators running IsiDetector on the `fps` branch.

---

## `./up.sh` seems to hang

**Symptom:** the terminal sits at "Waiting for web container to finish ONNX preload" for more than 60 seconds.

**What `up.sh` is actually waiting for:**

| Compose mode | Readiness signal |
|---|---|
| GPU | `ONNX preload (CUDA kernels warm` **or** `Running on http://` **or** `Uvicorn running on` **or** `Application startup complete` |
| CPU | `Running on http://` **or** `Uvicorn running on` **or** `Application startup complete` |

On a **GPU host**, the full cold-start takes ~30–35 s: the rfdetr sidecar imports its deps (~25 s), the web container waits for it to become healthy, then CUDA kernel autotuning runs (~5 s). This is normal.

**If it truly stalls past 90 s:**

```bash
# Check what the web container is actually doing
docker compose logs web | tail -30
```

Common causes and fixes:

| What you see in the logs | Fix |
|---|---|
| `No such file or directory: isidet/models/...` | The weight path in `settings.json` points to a file that doesn't exist. Drop the file into `isidet/models/` or clear the path in Settings. |
| `could not select device driver "nvidia"` | GPU marker is stale — see [GPU host falling back to CPU](#gpu-host-falling-back-to-cpu) below. |
| `Application startup complete` already in logs but terminal is still waiting | The FastAPI/uvicorn banner matches `Application startup complete` exactly (lowercase). If your image is old it may print a slightly different string. Rebuild: `docker compose build && ./up.sh`. |
| Nothing — logs empty | Container crashed at import time. Run `docker compose logs web` (no `-f`) to see the full output including the traceback. |

!!! note "FastAPI vs Flask banner"
    When `WEB_BACKEND=fastapi` is set, uvicorn prints `Uvicorn running on http://` and `Application startup complete`. Both are matched by `up.sh`. If only `Running on http://` was matched (old logic), FastAPI would always stall for the full `TIMEOUT_SEC`. If you see this, rebuild the image.

---

## Model dropdown is empty

**Symptom:** Settings page loads but the YOLO and/or RF-DETR weight dropdowns show nothing.

**Diagnosis:**

```bash
curl -s http://localhost:9501/api/models | python3 -m json.tool
```

If the response is `{"yolo": [], "rfdetr": []}`, the auto-discovery walk found no files.

**Fixes:**

1. **Weight files are missing.** Drop them into `isidet/models/` on the host (bind-mounted into the container — no restart needed):
   ```bash
   ls ~/fps/isidet/models/yolo/
   ls ~/fps/isidet/models/rfdetr/
   ```
   Discovery walks `isidet/models/yolo/**/weights/` for `.pt`, `.onnx`, `.xml`, `.engine` and `isidet/models/rfdetr/**/` for `.pth`, `.onnx`.

2. **Wrong directory.** You are in `~/logistic` but the stack is running from `~/fps` (or vice versa). Check which stack is running:
   ```bash
   docker compose ps           # from ~/fps
   ```
   The `fps` branch runs under compose project name `fps`; the deploy branch runs under `deploy`. Weights must be placed under the active clone's `isidet/models/`.

3. **Container not seeing new files.** The volume mount is live — no restart needed. If the API still returns an empty list after placing files, restart the web container to force a rescan:
   ```bash
   docker compose restart web
   ```

---

## Sorter receives no UDP triggers

**Symptom:** inference runs and detections appear in the UI, but the sorting controller logs no datagrams.

Default UDP target on `fps`: **`10.0.0.1:9502`**. UDP payload: `{"class": "carton", "seq": 42, "id": 7, "ts": "..."}` (gap-free monotonic `seq`).

**Step-by-step diagnosis:**

```bash
# Step 1 — check what the stack is actually targeting
curl -s http://localhost:9501/api/udp
# → {"host": "10.0.0.1", "port": 9502}
```

If the host/port is wrong, fix it live in Settings → UDP (no restart needed) or:

```bash
curl -X POST http://localhost:9501/api/udp \
     -H "Content-Type: application/json" \
     -d '{"host": "10.0.0.1", "port": 9502}'
```

```bash
# Step 2 — run net.sh test (includes a live UDP egress probe)
./net.sh test
```

The test output tells you exactly which hop stops the packet. Common findings:

| `net.sh test` output | Meaning | Fix |
|---|---|---|
| Step 5 ✅ (UDP sent, reply received) | End-to-end path works; problem is on the controller side | Check controller firewall / listener port |
| Step 5 ❌ (sent, no reply) | Packet leaves the PC but the controller doesn't answer | Controller firewall or wrong target IP |
| Step 4 ❌ (gateway unreachable) | Network not frozen or NIC on wrong subnet | Run `sudo ./net.sh setup` or `sudo ./net.sh apply` |
| Step 3 ❌ (web container not running) | Stack is down | `./up.sh` |

```bash
# Step 3 — watch UDP publish latency in real time
curl -s http://localhost:9501/api/performance | python3 -m json.tool | grep -A5 udp
```

If `datagrams_sent` is 0 and inference is running, check the stream is actually crossing the line zone. Move the line in Settings or trigger a test crossing manually.

!!! warning "net.sh needs NetworkManager"
    `net.sh` only works on Ubuntu Desktop with NetworkManager. It will exit cleanly with "NetworkManager not installed" on WSL2 or Ubuntu Server. On those hosts, verify UDP egress manually with `nc -u 10.0.0.1 9502`.

---

## GPU host falling back to CPU

**Symptom:** `up.sh` prints `⚠ .deployment.env says GPU but nvidia-smi is unavailable` and starts the CPU stack instead.

**Cause:** `.deployment.env` contains `COMPOSE_MODE=gpu` from a previous install, but the NVIDIA driver is no longer visible — driver update, kernel upgrade, or the PC was migrated from a GPU dev box.

**Fix:**

```bash
# Verify the driver is reachable on the host
nvidia-smi

# If that fails, reinstall or reload the driver, then recheck.
# Once nvidia-smi works, refresh the deployment marker:
cd ~/fps
./run_start.sh              # re-detects hardware, rebuilds, rewrites .deployment.env
```

If the host genuinely has no GPU and you want to run CPU permanently:

```bash
./up.sh --force-cpu
# Then make it permanent:
echo "COMPOSE_MODE=cpu" > deploy/.deployment.env
```

!!! note "Intentional fallback"
    The fallback is a safety net — it prevents the cryptic `could not select device driver "nvidia" with capabilities: [[gpu]]` error when hardware changes. Pass `--force-gpu` explicitly only if you are certain the driver is present and want to bypass the check.

---

## `settings.json` git pull conflict

**Symptom:** `git pull` on `~/fps` fails with a merge conflict in `webapp/isitec_app/settings.json` (or the FastAPI equivalent).

**Cause:** the on-site file has operator-specific values (RTSP URL, line config, UDP host) that diverge from what upstream committed.

**One-time fix — set skip-worktree (do this once per site PC):**

```bash
cd ~/fps
git update-index --skip-worktree \
    webapp/isitec_app/settings.json \
    webapp/isitec_api/settings.json
```

After that, `git pull` leaves both files untouched forever. New upstream keys appear automatically the next time the operator opens Settings and clicks Save (the backend merges them on write).

**If you already have a conflict right now:**

```bash
cd ~/fps
# 1. Back up operator settings
cp webapp/isitec_app/settings.json /tmp/settings_backup.json
cp webapp/isitec_api/settings.json /tmp/settings_api_backup.json

# 2. Accept upstream version to clear the conflict
git checkout -- webapp/isitec_app/settings.json
git checkout -- webapp/isitec_api/settings.json

# 3. Pull cleanly
git pull

# 4. Restore operator values
cp /tmp/settings_backup.json webapp/isitec_app/settings.json
cp /tmp/settings_api_backup.json webapp/isitec_api/settings.json

# 5. Set skip-worktree so it never happens again
git update-index --skip-worktree \
    webapp/isitec_app/settings.json \
    webapp/isitec_api/settings.json
```

Then rebuild so the new code picks up the operator's restored settings:

```bash
./up.sh
```

---

## CLAHE hurting carton counts

**Symptom:** after enabling SpecularGuard/CLAHE (Settings → Performance → CLAHE), carton detection rate drops or class predictions flip between carton and polybag.

**Why it happens:** CLAHE enhances local contrast on the LAB L-channel. It is designed for polybag glare suppression. On matte-surface cartons (already high local contrast), the enhancement can push textures outside the training distribution and confuse the classifier head at 320 px input.

**Fix:**

1. Disable CLAHE in Settings → Performance — tick off CLAHE / SpecularGuard → Save → Stop and Start the stream.
2. Verify counts recover. CLAHE default is **OFF**; only enable it if polybag glare is visibly causing missed detections.
3. If the belt has mixed cartons and polybags with strong glare, try increasing confidence threshold (`yolo_conf`) to 0.60–0.65 rather than enabling CLAHE globally.

!!! tip "ROI crop often helps more than CLAHE"
    Cropping the belt region before inference (Settings → Camera → Show Set ROI button) raises effective pixel density for both classes without altering colour statistics. It is a better first response to classification flips than CLAHE. See [Site Operations](site-operations.md#4-roi-crop--live-page-4-click-belt-configurator) for the setup flow.
