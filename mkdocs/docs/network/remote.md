# Remote Access (remote.sh)

Set up, check, and tear down unattended remote access to a site PC using Tailscale (VPN mesh) and RustDesk (remote desktop).

---

## How it works

Two services work together:

| Service | Role |
|---|---|
| **Tailscale** | Zero-config VPN mesh. Gives the admin laptop and the site PC a shared `100.x.x.x` address space — no port-forwards, no public IP required. |
| **RustDesk** | Remote desktop. Once the network is up, connects the admin to the kiosk screen over Tailscale or the local LAN. |

RustDesk requires an **X11 session**. `remote.sh setup` detects GDM3/Wayland and disables Wayland automatically. A reboot is required if the current session is Wayland — the script tells you.

---

## Commands

```bash
./remote.sh              # same as status
./remote.sh setup        # install + configure both services (auto-escalates to sudo)
./remote.sh status       # show Tailscale IP and RustDesk ID
./remote.sh test         # connectivity probes (read-only, no sudo needed)
./remote.sh remove       # full uninstall + cleanup
./remote.sh --help
```

`setup` and `remove` need root and re-exec themselves via `sudo -E` if not already running as root.
`status` and `test` are read-only and run as the regular user.

---

## Setup

### Standard (interactive SSO)

```bash
./remote.sh setup
```

1. The script prints a Tailscale login URL.
2. Open the URL in Chrome on the kiosk and sign in with the Gmail account that owns the tailnet.
3. If your tailnet requires device approval, approve the device at <https://login.tailscale.com/admin/machines>.
4. Return to the terminal. The script polls for the IP automatically (5-minute timeout). Press **Enter** to check immediately.
5. RustDesk is installed and configured with the fleet password (see below).
6. If the script prints **REBOOT REQUIRED**, run `sudo reboot` before connecting.

### Unattended (auth key + custom password)

```bash
sudo ./remote.sh setup --ts-key tskey-auth-... --rd-password 'SomeOtherPW'
```

`--ts-key` skips the browser flow entirely — suitable for scripted provisioning.
`--rd-password` overrides the fleet default for this host only.

---

## Fleet RustDesk password

The default permanent password set on every site PC is:

```
Isitec69+
```

This is intentional — a single credential the admin team remembers across all sites. Tailscale is the access perimeter; the RustDesk password is the second layer at the session level.

To rotate the password on an already-deployed site PC:

```bash
su - <desktop_user> -c "rustdesk --password 'NEW_PW'"
sudo systemctl restart rustdesk
```

Changing `RD_DEFAULT_PASSWORD` in the script only affects fresh installs.

---

## Connecting from the admin laptop

After setup, the script prints a summary. Three connection methods are available from your RustDesk client:

| Method | When to use | What to enter |
|---|---|---|
| **ID + password** | Anywhere with internet | RustDesk ID, then `Isitec69+` |
| **Direct IP via Tailscale** | Preferred — fast, no public relay | `100.x.x.x:21118`, then password |
| **Direct IP via LAN** | On-site, no Tailscale needed | `<lan-ip>:21118`, then password |

Direct IP uses TCP port **21118** (RustDesk default).

SSH over Tailscale is also enabled if the tailnet ACL allows it:

```bash
tailscale ssh <user>@<tailscale-ip>
```

---

## Status

```bash
./remote.sh status
```

Shows:

- Display manager and current session type (X11 vs Wayland)
- Tailscale connection state and `100.x.x.x` IP
- RustDesk service state and device ID
- Last setup details from `/var/log/isidetector/remote-state.json`

To read the state file directly (contains the RustDesk ID and plaintext password):

```bash
sudo cat /var/log/isidetector/remote-state.json
```

---

## Test

```bash
./remote.sh test
```

Runs five read-only probes:

1. Internet reachability
2. Session type (X11 pass / Wayland warn)
3. Tailscale connected + IP
4. Tailnet peer reachability on TCP 22 (non-fatal)
5. RustDesk service running

---

## Wayland / X11 note

RustDesk screen capture and input injection do not work on Wayland. `remote.sh setup` disables Wayland in `/etc/gdm3/custom.conf` for GDM3 hosts. LightDM and SDDM default to X11 and are left unchanged.

!!! warning "Reboot required after Wayland disable"
    If the script reports **REBOOT REQUIRED**, the config change is written but the current session is still Wayland. Reboot before attempting a RustDesk connection or you will see a black screen.

    ```bash
    sudo reboot
    ```

!!! note "GDM3 backup"
    The original `custom.conf` is backed up to `/etc/gdm3/custom.conf.pre-remote-<timestamp>` before any edit. `./remote.sh remove` restores it automatically.

---

## Remove

```bash
./remote.sh remove
```

Type `remove` at the confirmation prompt. This:

- Logs out the device from the tailnet and purges the Tailscale package and state.
- Stops and purges RustDesk; wipes all per-user config directories (a fresh `setup` will get a new RustDesk ID).
- Restores `/etc/gdm3/custom.conf` from the backup if one exists.
- Deletes the state file at `/var/log/isidetector/remote-state.json`.

!!! tip "Re-provisioning"
    After `remove`, run `./remote.sh setup` again to get a clean install with a new RustDesk ID.
