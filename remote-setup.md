# 🌐 Remote Access — How to set up a new site PC

One-page checklist. Follow the four steps in order. ~10 minutes including the Tailscale browser sign-in.

---

## What you get

- **Tailscale**: site PC joins your private mesh at a `100.x.x.x` IP. Reachable from your laptop anywhere with internet, no port-forwards.
- **RustDesk**: full remote desktop GUI. Connects via Tailscale (preferred), LAN, or public RustDesk relay.
- **Fleet password**: `Isitec69+` on every site PC by default.

---

## Prerequisites

- Site PC: Ubuntu/Debian with internet (apt + GitHub reachable).
- The repo cloned at `~/logistic` on the site PC, on the `fps` branch.
- A Gmail account that owns the Tailscale tailnet (or a Tailscale auth key).

---

## Step 1 — Wipe any prior state (always start here)

Even on a brand-new PC, this guarantees no half-installed leftovers from a previous attempt.

```bash
cd ~/logistic && git pull origin fps
sudo ./remote.sh remove
# Type 'remove' when prompted.
```

Wait for the "system is CLEAN" line at the end. If you see `[WARN] xyz still present`, run `sudo apt purge -y tailscale rustdesk && sudo apt autoremove -y` and re-run `./remote.sh remove`.

---

## Step 2 — Confirm the system is clean

```bash
./remote.sh status
```

You should see:
```
Tailscale: not-installed
RustDesk:  not-installed
```

If anything else shows, repeat Step 1.

---

## Step 3 — Install + configure

```bash
sudo ./remote.sh setup
```

What happens:
1. Pre-flight check (internet, distro, required tools).
2. **Tailscale** installs. A login URL prints — open it on the kiosk's Chrome, sign in with Gmail. Approve the device on the admin dashboard if your tailnet requires it. The script auto-detects when authentication completes.
3. **GDM3** is reconfigured to use X11 instead of Wayland (RustDesk requires X11). **Reboot is needed after setup** for this to take effect.
4. **RustDesk** installs, the systemd service starts, options are written (permanent password mode, direct IP access on port 21118), and the password `Isitec69+` is set.
5. Final summary prints **all connection details**.

If you see a `REBOOT REQUIRED` warning at the end:
```bash
sudo reboot
```

---

## Step 4 — Verify and record

After reboot (or immediately if no reboot was needed):

```bash
./remote.sh status
```

Expected output:
```
Display:   gdm3 | session: x11

Tailscale: connected
  IP:      100.x.x.x
  Peers:   ...

RustDesk:  running
  ID:      <9-digit ID>

Last setup state (from /var/log/isidetector/remote-state.json):
  {
    "tailscale": { "ip": "100.x.x.x", ... },
    "rustdesk":  { "id": "...", "password": "Isitec69+", ... }
  }
```

📸 **Take a photo of the screen now.** You need the four values:
- **Tailscale IP** (`100.x.x.x`)
- **LAN IP** (`192.168.x.x`)
- **RustDesk ID** (9 digits)
- **RustDesk password** (`Isitec69+` unless overridden)

---

## Connect from your laptop

Open RustDesk on your laptop. In the "Remote ID" field at the top, paste **one** of the following:

| Method | What to enter | When to use |
|---|---|---|
| **Direct via Tailscale** (preferred) | `100.x.x.x:21118` | From anywhere — fast, private, no public relay |
| **Direct via LAN** | `192.168.x.x:21118` | When physically on-site |
| **9-digit ID** (fallback) | `<id>` | If direct IP fails for any reason |

Then enter password `Isitec69+` (or your override).

For SSH access:
```bash
tailscale ssh <site_user>@100.x.x.x
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Tailscale auth times out | Check device-approval at https://login.tailscale.com/admin/machines |
| `tailscale binary still present` after remove | Re-run `sudo ./remote.sh remove` (sweeps leftover binaries) |
| RustDesk asks for a different password | Verification mode wasn't set. Re-run `sudo ./remote.sh setup` |
| Black screen / no input via RustDesk | Site PC is on Wayland. Reboot, or verify `./remote.sh status` shows `session: x11` |
| Lost the password | `sudo cat /var/log/isidetector/remote-state.json` |

---

## Per-site overrides

```bash
# Different password for this site
sudo ./remote.sh setup --rd-password 'CustomPW'

# Tailscale auth key (skip the browser sign-in)
sudo ./remote.sh setup --ts-key tskey-auth-XXXXXXXXX

# Self-hosted RustDesk relay
sudo ./remote.sh setup --rd-server self.example.com
```
