# autostart.sh — remove kiosk mode, become a headless boot-time stack installer

**Date:** 2026-07-30
**Branch:** `fps` (site PCs run this branch)
**Status:** approved design, pending implementation

## Problem

The three-layer kiosk installer (`autostart.sh`) proved problematic on site: Layer 1
edits display-manager config (`/etc/gdm3/custom.conf`: `AutomaticLogin*`,
`WaylandEnable=false`) and Layer 3 auto-launches fullscreen kiosk Chrome. The
display-manager edits risk breaking the login screen / graphics session
(Wayland vs X11), and because the operator avoided enabling autostart, a PC
reboot left detection stopped until someone opened the browser and clicked
Start — which also starved the flow recorder of events.

## Goal

Boot must survive Chrome close, Docker restart, and full PC restart with **zero
display-manager involvement**:

- power on → systemd brings up `docker compose` → container's `auto_start`
  replays the last successful Start → detection + UDP + event CSV running headless.
- No auto-login, no kiosk Chrome, no `custom.conf` edits. The operator opens a
  browser manually only when they want to watch.
- `remote.sh` remains the sole owner of the Wayland/X question (RustDesk capture).

## Design

Approach chosen: **rewrite `autostart.sh` in place** — same filename, same
`enable/disable/status` subcommands, same root thin wrapper — but single-layer.
All display-manager *install* code is deleted; only legacy *cleanup* code may
reference it.

### 1. Script surface

- **`enable [USER]`** (auto-escalates to sudo):
  1. Legacy cleanup (§2).
  2. Install `/etc/systemd/system/isidetector.service` — unit body unchanged
     from today: `Type=oneshot` + `RemainAfterExit`, `WorkingDirectory=$INSTALL_DIR`,
     `User=` install-dir owner, `ExecStart=docker compose up -d`,
     `ExecStop=docker compose down`, `After=docker.service network-online.target`.
     Then `daemon-reload` + `enable` + `start`.
  3. Flip `auto_start=true` (§3).
  4. Print end-state summary, including: the **first-ever** hands-free boot
     still requires one manual Start click beforehand, so `last_model_type` /
     `last_weights` get recorded for `auto_start` to replay.
  - `USER` arg survives only to resolve legacy artifact home dirs and the
    settings-file owner fallback.
- **`disable`**: stop/disable/remove the unit + the same legacy cleanup.
  `auto_start` is left untouched (harmless without the boot unit; operator state).
- **`status`** (read-only, no sudo): unit installed/enabled/active state,
  current `auto_start` value from settings.json, a
  `⚠ legacy kiosk artifacts found` warning when applicable, then the existing
  `_run_diagnostics` block. The diagnostics' Wayland line becomes informational
  (points at `remote.sh`), no longer advises rebooting via autostart.
- Header docstring rewritten to the headless single-layer story; `--help` keeps
  printing the header.

### 2. Legacy cleanup (shared by `enable` and `disable`)

Cleans up installs made by the old three-layer script. Idempotent; silent when
there is nothing to do; prints each artifact it removes.

- Remove `isidetector.desktop` from every `/home/*/.config/autostart` and
  `/root/.config/autostart`.
- GDM3 (`/etc/gdm3/custom.conf`):
  - newest `.pre-autostart-*` backup exists → restore it (recovers the
    pre-kiosk file including any sysadmin keys);
  - no backup → surgically delete `AutomaticLogin*` lines only.
    **`WaylandEnable` is never touched in this path** — that key is co-owned by
    `remote.sh` (RustDesk needs X11) with its own `.pre-remote-*` backups.
- LightDM / SDDM: delete our
  `50-isidetector-autologin.conf` drop-ins if present.

### 3. `auto_start` flip

- Stack answering on `tcp/9501` → `POST /api/settings {"auto_start": true}` so
  the running backend applies it immediately.
- Otherwise → edit `webapp/isitec_app/settings.json` and
  `webapp/isitec_api/settings.json` directly (whichever exist) with a
  `python3` JSON load/set/dump one-liner; `chown` each file back to the
  install-dir owner. No other keys touched.
- Any failure here is a **warning, not fatal** — the systemd install is never
  rolled back because a settings write failed.

### 4. Docs, safety, verification

- Update `start.md` and `CLAUDE.md` (this branch) `autostart.sh` sections to the
  single-layer story.
- Grep-verifiable invariant: `gdm3` / `WaylandEnable` / `custom.conf` appear
  **only** inside the legacy-cleanup function (plus diagnostics read-only lines).
- Dev-box verification (WSL2, no systemd DM): `bash -n`, shellcheck, dry-run of
  the settings-flip python against a scratch settings.json, `status` end-to-end
  (probes already degrade gracefully). Real end-to-end verification happens on
  the site PC after `git pull`.
- Lands on `fps` as a normal commit (not an experimental-prefixed one).

## Out of scope

- `up.sh` keeps its `--kiosk` / `--open-only` flags (manual use only; nothing
  auto-invokes them anymore).
- `remote.sh` unchanged.
- The stuck-ffmpeg flow-recorder hardening (`rtsp_record_30min.sh` timeouts) is
  a separate change.
