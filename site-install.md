# 🛠️ Site Install — Field Engineer Runbook

**Mission:** turn a fresh hardware bundle into an isi-linux PC that's streaming from the camera and remotely reachable from the office. ~60–90 minutes total.

**You are done when:** the office team confirms over Tailscale that they can see the live stream on their laptop.

---

## Pre-visit

📦 Verify the case against [`list.md`](list.md) before leaving the office. Critical items: site PC, USB-to-ethernet adapter, RJ45 splitter, RJ45 tester, RJ45 patch cables (short + long), camera + bracket + PSU, the printed copy of this runbook.

---

## Network reference (memorise — you'll use it constantly)

```
                ┌─ Splitter ──┐
  Customer wall ─┤             ├──→  PC Superviseur (192.168.1.20)
   internet      │             ├──→  ISI-Linux enp2s0   (DHCP, internet)
                └─────────────┘

  ISI-Linux enp1s0   ── 192.168.1.50/24 ──→  Camera (192.168.1.108)        [Réseau Caméra]
  ISI-Linux USB-eth  ── 10.0.0.5/24    ──→  Internal switch ──→ AutoMate (10.0.0.10)  [Réseau Automate]
```

| Interface (ISI-Linux) | IP / mask        | Network              | Purpose                         |
|---|---|---|---|
| `enp2s0` (onboard NIC 2) | DHCP             | Réseau Internet      | uplink for updates, Tailscale, RustDesk |
| `enp1s0` (onboard NIC 1) | 192.168.1.50/24  | Réseau Caméra        | RTSP stream from the camera     |
| `enx…` (USB-to-Ethernet) | 10.0.0.5/24      | Réseau Automate      | UDP sort triggers to PLC        |

⚠️ **Do not connect the camera or automate cables yet.** Step 1 first.

---

## 1️⃣  Splitter — share the customer's internet cable

The customer hands you **one** ethernet cable from the wall. Plug your splitter into it; route one leg to the **PC Superviseur** and the other leg to the **ISI-Linux PC** (left for now — we'll connect it in step 2b). Power both PCs on.

📸 **Photo 1:** the splitter with both legs visible, plus the customer's wall jack.

---

## 2️⃣  ISI-Linux network — three NICs, in this exact order

The order matters: each NIC is configured one at a time so `nmcli` creates clean per-cable profiles.

### 2a · Clean slate

Unplug **every** ethernet cable from the back of the ISI-Linux PC. Open a terminal.

```bash
ip -o link show | awk -F': ' '{print $2}' | grep -vE '^lo$'
```

Note the interface names that appear (typically `enp1s0`, `enp2s0`, and after step 2d an `enx…` adapter).

### 2b · Internet uplink (DHCP)

Plug **only** the internet cable from the splitter into onboard NIC 2 (`enp2s0`). Then:

```bash
cd ~/logistic && git pull origin fps
sudo ./net.sh setup
```

Walk through the prompts: for `enp2s0` choose **`dhcp`**, leave gateway/DNS blank (DHCP supplies them). Skip every other interface for now.

Verify:

```bash
ip -4 addr show enp2s0   # expect a 192.168.x.x or similar from the customer's subnet
ping -c 3 1.1.1.1        # expect 3/3 reply, latency < 100 ms
```

If the ping fails, check the splitter or re-plug the wall cable. Do **not** continue until DHCP works.

### 2c · Camera network (static)

Plug the camera cable into onboard NIC 1 (`enp1s0`). Leave the camera unpowered for now.

```bash
sudo ./net.sh setup
```

For `enp1s0` choose **`static`**, IP **`192.168.1.50`**, mask **`24`**, gateway **blank**, DNS **blank**. Skip the others.

Verify:

```bash
ip -4 addr show enp1s0   # expect 192.168.1.50/24
```

### 2d · Automate switch (USB-to-Ethernet)

Plug the **USB-to-ethernet adapter** into a free USB port on the ISI-Linux PC. Plug the short patch cable from that adapter into the internal switch. Re-run:

```bash
ip -o link show | grep -E '^[0-9]+: enx'   # find the new adapter name
sudo ./net.sh setup
```

For the `enx…` interface choose **`static`**, IP **`10.0.0.5`**, mask **`24`**, gateway **blank**, DNS **blank**.

Verify (the AutoMate may not be powered yet — link state is what matters):

```bash
ip -4 addr show          # all three NICs visible with the right IPs
ip link show             # confirm enx… is "state UP"
ping -c 3 10.0.0.10      # may fail if the PLC is off — that's OK
```

📸 **Photo 2:** the back of the PC with all three cables labelled (Cam / Net / Automate). 📸 **Photo 3:** terminal showing the full `ip -4 addr show` output.

---

## 3️⃣  Camera install + smoke test

Mount the camera per the installer's spec, route the cable, power it up. Wait ~30 seconds for it to boot.

```bash
ping -c 3 192.168.1.108
```

3/3 reply means the camera's reachable. If not, check power LED + cable + try `sudo nmcli con up "Cam"`.

Open Chrome on the ISI-Linux PC → `http://192.168.1.108`. Log in (default creds usually `admin / admin` — check the camera's sticker). Confirm you see the live preview.

Then run the project's RTSP probe:

```bash
cd ~/logistic
./cam_status.sh
```

Expected: `[OK] reachable` + a line like `📹 Stream: 1920×1080 @ 25 fps codec=h264`. If FPS is < 20, note the value — the camera may be the FPS ceiling and we'll switch to a sub-stream URL later.

📸 **Photo 4:** the camera mounted, with the belt visible behind it.
📸 **Photo 5:** the terminal showing `cam_status.sh` output.

---

## 4️⃣  Remote access — Tailscale + RustDesk

Now follow the four steps in [`remote-setup.md`](remote-setup.md) **verbatim**. When the Tailscale browser auth URL opens, sign in with the **`isivision`** Gmail account (the office team will give you the password if you don't have it).

After `setup` completes (and the PC reboots if prompted):

```bash
./remote.sh status
```

📸 **Photo 6:** the full `./remote.sh status` output — must show:
- `Display: gdm3 | session: x11`
- `Tailscale: connected` + the `100.x.x.x` IP
- `RustDesk: running` + the 9-digit ID
- The state-file dump at the bottom (with the password — fleet default is `Isitec69+`)

---

## 5️⃣  Office handoff

Send all six photos to the office team (WhatsApp + email both, in case one fails). The office team will:

1. Connect to the ISI-Linux PC via Tailscale (`tailscale ssh` + RustDesk to `100.x.x.x:21118`).
2. Validate the live inference stream end-to-end.
3. Confirm sort-trigger UDP datagrams reach the AutoMate at `10.0.0.10:9502`.

Stand by on-site for ~15 minutes in case a cable needs re-seating. The office will message you "we're in, you can leave." Once you get that message, you're done — pack up.

---

## 🆘 Troubleshooting

| Symptom | First thing to try |
|---|---|
| `enp2s0` doesn't get a DHCP IP | re-plug the splitter; check the customer's wall jack with the RJ45 tester |
| `ping 192.168.1.108` fails | camera not yet powered, or wrong cable; `sudo nmcli con up "Cam"` after re-plugging |
| `cam_status.sh` says "no stream" | open the camera's web GUI → System → RTSP — confirm enabled, port 554 |
| Tailscale auth never completes | device awaiting approval — call the office, they'll click Approve at https://login.tailscale.com/admin/machines |
| `./remote.sh status` shows `session: wayland` | reboot the PC once and re-check (the X11 switch needs a reboot) |
| `apt purge` or install fails during `remote.sh setup` | `sudo dpkg --configure -a && sudo apt-get -f install`, then re-run setup |
| Camera FPS stuck < 20 | switch the camera URL to a sub-stream (`rtsp://…/2` instead of `…/1`) via Settings → Camera in the dashboard — office can do this remotely |

If you hit anything not in this table, photo the terminal output and call the office before continuing.

---

## What you should NOT do

- Don't edit `/etc/netplan/*.yaml` by hand — `net.sh setup` writes proper NetworkManager profiles that survive reboot.
- Don't run `ip addr add 192.168.1.50/24 dev enp1s0` — that doesn't persist.
- Don't skip the photos — they're the office team's only way to verify your work without being on-site.
- Don't connect the AutoMate cable before configuring the USB adapter — the customer's PLC may have an aggressive ARP that confuses Linux's interface init if it sees an IP it doesn't expect.

---

📄 **Pair this with** [`remote-setup.md`](remote-setup.md) (remote-access detail) and [`list.md`](list.md) (packing checklist).
