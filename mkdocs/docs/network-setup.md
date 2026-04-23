# Network Setup — `net.sh`

Simple helper that fixes the two things that usually break UDP between the site PC and the automate.

## What it does

1. **The site PC's IP keeps changing (DHCP)** → `apply` freezes it.
2. **You don't know if UDP is actually leaving the PC** → `test` proves it.

It also prints a ready-to-email guide for the automation engineer on the other side.

## Steps — in order

### Step 1 — Look at what you have now

```bash
./net.sh show
```

Shows IP, gateway, DNS, the automate IP/port, and whether you're on DHCP (⚠) or locked (✅).

### Step 2 — Freeze the IP so it never changes again

```bash
sudo ./net.sh apply
```

Takes the current working IP/gateway/DNS and makes them permanent. After a reboot, Wi-Fi drop, or lease expiry, the PC always comes back on the same IP. This is what the automaticien needs.

### Step 3 — Test that UDP actually gets out

```bash
./net.sh test
```

Runs 5 checks:

1. Can I reach the gateway?
2. Can I reach the internet?
3. Can I reach the automate?
4. Is the web container publishing to the right address?
5. **Live probe** — sends a real UDP packet from inside Docker and sniffs the network card to confirm it left the PC.

If all 5 are ✅ → UDP is fine. If any ❌ → that's your exact problem.

### Step 4 — Send the guide to the automation engineer

```bash
./net.sh manual          # French
./net.sh manual --en     # English
```

Prints a copy-paste manual with your IP, their IP/port, the JSON format, firewall rules, and a handshake test they can run on their side.

### Step 5 — Undo everything (if needed)

```bash
sudo ./net.sh revert
```

Goes back to DHCP.

## Normal workflow on a new site

```bash
./net.sh show          # see the state
sudo ./net.sh apply    # freeze the IP
./net.sh test          # prove UDP leaves the PC
./net.sh manual        # email the output to the automaticien
```

Four commands — the network side is done.
