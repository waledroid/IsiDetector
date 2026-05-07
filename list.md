# Field Install Checklist — CPU-only site

Quick grab-bag. Tick each box as it goes in the case.

## PC
- [ ] Site PC (Intel CPU-only, Ubuntu 22.04+ with Docker installed)
- [ ] PC power cable
- [ ] Monitor + HDMI/DP cable (or tiny portable screen)
- [ ] Keyboard + mouse (wired — Wi-Fi KB can't log in during first-boot)
- [ ] Short Ethernet cable (PC → wall jack / switch)

## Camera
- [ ] IP camera (RTSP-capable, industrial grade)
- [ ] Lens suited to the belt distance (check FoV before leaving)
- [ ] PoE injector **OR** 12 V PSU (whichever the camera needs)
- [ ] Mounting bracket + bolts/nuts for the belt frame
- [ ] Ethernet cable for the camera run (measure twice — bring spare)

## Network
- [ ] 2–3 spare Ethernet patch cables (short + long)
- [ ] Small unmanaged switch (5-port) in case the sorter/camera/PC need to share a segment
- [ ] RJ45 cable tester
- [ ] Your laptop (for SSH, curl, `./net.sh manual` printouts)

## Power
- [ ] Power strip with surge protection
- [ ] Extension cord (≥ 5 m)

## Tools
- [ ] Screwdrivers (Phillips + flat, small + medium)
- [ ] Adjustable wrench / spanner
- [ ] Cable ties + scissors
- [ ] Electrical tape + masking tape
- [ ] Sharpie for labels
- [ ] Flashlight (headlamp beats handheld)
- [ ] Multimeter (check voltage on the camera PSU)

## Software (on a USB stick, just in case)
- [ ] Ubuntu 22.04 Live USB (bootable)
- [ ] `deploy` branch cloned locally (offline copy)
- [ ] Docker CPU image tarball as fallback (`docker save isidet-cpu > isidet-cpu.tar`)
- [ ] SSH key to pull updates from GitHub

## Docs (printed)
- [ ] `site-install.md` — **the runbook**, follow it step-by-step on the day
- [ ] `remote-setup.md` — the four-step Tailscale + RustDesk reference
- [ ] `start.md` from the `deploy` branch (install → daily start → net lock-down)
- [ ] `net.sh manual` output — bilingual FR/EN sheet for the automaticien
- [ ] Blank form for sorter UDP target + camera RTSP URL + line settings

## To capture on site
- [ ] Sorter controller **IP + UDP port** → set via `POST /api/udp` or env
- [ ] Camera **RTSP URL + credentials**
- [ ] Line orientation / belt direction / line position (eyeballed at install)
- [ ] Network details: PC IP, gateway, DNS — freeze with `./net.sh apply`
- [ ] Photos of: camera mount, PC rack, network jack labels

## Safety
- [ ] Safety shoes, hi-vis vest, safety glasses
- [ ] Site access badge / supervisor contact

## First-boot sanity (run on site before leaving)
1. `./run_start.sh` → Docker + CPU image build
2. `./up.sh` → stack comes up, browser opens `:9501`
3. Set model, RTSP source, line, belt direction in the UI
4. `./net.sh test` → confirm UDP egress to the sorter is reachable
5. `./net.sh apply` → freeze DHCP into static NM config
6. Hand over `start.md` printout to the on-site operator
