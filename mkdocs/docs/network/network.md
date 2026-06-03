# Network Lock-down (net.sh)

Freeze the site PC's DHCP-issued IP as a permanent static address and verify that UDP sort triggers are leaving the machine.

---

## Why this matters

The automate's firewall whitelists the site PC by IP. If the PC gets a new DHCP lease after a reboot or Wi-Fi drop, UDP stops working. `net.sh apply` writes the current IP into NetworkManager permanently so it never drifts.

!!! warning "NetworkManager only"
    `net.sh` requires **Ubuntu Desktop** with NetworkManager (`nmcli`). It will refuse with a clear error on WSL2, Ubuntu Server (which uses `systemd-networkd`/netplan), or any machine without `nmcli`. Run it on the actual site PC.

---

## Commands at a glance

| Command | Needs sudo | What it does |
|---|---|---|
| `./net.sh show` | No | Print current IP, gateway, DNS, lock status, and UDP publisher state |
| `./net.sh apply` | Auto-escalates | Freeze DHCP values as static; bounce the connection |
| `./net.sh revert` | Auto-escalates | Return to DHCP |
| `./net.sh test` | Auto-escalates | Run 5 reachability + live UDP egress checks |
| `./net.sh setup` | Auto-escalates | Interactive multi-NIC wizard (dual-NIC site PCs) |

`apply`, `revert`, `test`, and `setup` automatically re-execute themselves with `sudo -E` if you are not already root — you do not need to prefix them manually (though doing so is harmless).

---

## Step-by-step: standard single-NIC site

### 1. Check the current state

```bash
./net.sh show
```

Output includes:

- Active NM connection name, type, and interface
- IPv4 method — `manual` (locked, green) or `auto` (DHCP, yellow warning)
- IP/mask, gateway, DNS
- Route to the automate controller
- UDP publisher target read live from the running web container

Run this first on every visit to confirm nothing has drifted.

### 2. Freeze the IP

```bash
./net.sh apply
```

Reads the live IP/gateway/DNS from NetworkManager, switches the profile to `ipv4.method manual`, sets `autoconnect=yes priority=100`, and bounces the connection. Re-runs `show` at the end so you can confirm the lock.

!!! tip "Override individual values"
    If the discovered values are wrong, pass overrides before writing:
    ```bash
    ./net.sh apply --ip 192.168.1.50/24 --gateway 192.168.1.1 --dns '8.8.8.8 8.8.4.4'
    ```
    Use `--conn NAME` to target a specific NM profile instead of the auto-discovered one.

Skip the confirmation prompt with `--force`:

```bash
./net.sh apply --force
```

### 3. Prove UDP is leaving the PC

Start the stack first (`./up.sh`), then:

```bash
./net.sh test
```

Five checks run in order:

1. **Gateway reachable** — ping the default gateway
2. **Internet reachable** — ping `1.1.1.1`; marked `skip` (not a failure) on air-gapped LANs
3. **Automate reachable** — ping the UDP controller IP read from the web container
4. **Web container publisher** — confirms `UDP_HOST` and `UDP_PORT` are set inside the container
5. **Live UDP egress** — sends a real UDP probe datagram from inside Docker and uses `tcpdump` on the NIC to confirm the packet actually left the machine

All five ✅ means the sort trigger path is clear end-to-end. Any ❌ identifies the exact broken link.

!!! note "UDP payload format"
    Each sort trigger datagram contains `class`, `seq`, `id`, and `ts`. `seq` is a gap-free monotonic counter. See [network/udp.md](udp.md) for the full protocol.

### 4. Undo (if needed)

```bash
./net.sh revert
```

Clears the static config and re-enables DHCP on the active connection.

---

## Multi-NIC site PCs (two LAN ports, no default gateway)

Some site PCs have two Ethernet ports — one toward the automate subnet, one toward a plant network — and no default gateway on either. Use the interactive wizard instead of `apply`:

```bash
./net.sh setup
```

The wizard:

1. Lists every physical NIC (`en*`/`eth*`/`wl*`) with its current link state, MAC, IP, and NM profile name
2. For each NIC, prompts: **[s]tatic / [d]hcp / s[k]ip**
3. For static NICs, asks for IP, CIDR prefix (default `/24`), gateway (blank = none), and DNS
4. Writes the config and bounces each connection immediately; a failed bring-up (cable unplugged) is saved and will auto-connect when the link comes up

After `setup`, run `./net.sh show` to verify both NICs are locked.

---

## Typical workflow on a new site

```bash
./net.sh show          # read current state
./net.sh apply         # freeze the IP
./net.sh test          # confirm UDP leaves the PC
```

For dual-NIC machines replace `apply` with `setup`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NetworkManager (nmcli) is not installed` | Running on WSL2 or Ubuntu Server | Run on the Ubuntu Desktop site PC |
| `No active non-virtual NetworkManager connection found` | Wi-Fi/Ethernet not connected | Connect first, then re-run |
| `Couldn't discover an IPv4 address` | NIC up but no IP yet | Pass `--ip A.B.C.D/PREFIX` explicitly |
| `Discovered IP is in Docker-bridge range` | Wrong NIC selected | Pass `--conn NAME` to target the correct profile |
| Test check 5 ❌ (UDP egress) | Container not running, firewall drop, or wrong `UDP_HOST` | Confirm `./up.sh` is running; check automate firewall whitelist |
| `Couldn't bring connection back up` after apply | Bad IP/gateway combo | Run `./net.sh revert`, correct the values, retry |
