#!/usr/bin/env bash
# ============================================================================
# IsiDetector — One-Shot Install Script for Production Inference PCs
#
# Target: fresh Ubuntu 22.04 / 24.04 LTS (native Linux, NOT WSL)
#
# What it does:
#   1. Checks the system meets prerequisites (sudo, apt, compatible Ubuntu).
#   2. Installs git + curl if they're missing.
#   3. Clones the IsiDetector repo (dev branch) into $INSTALL_DIR.
#   4. Makes the runtime scripts executable.
#   5. Optionally runs ./run_start.sh to bootstrap Docker + build the image.
#
# What it does NOT do:
#   - Does not install trained model weights. Those must be copied manually
#     into $INSTALL_DIR/models/ or $INSTALL_DIR/runs/ after install.
#     Typical: scp -r models/yolo/<date>/openvino user@prod-pc:~/logistic/...
#   - Does not set up auto-start on boot (see README note at the bottom of
#     run_start.sh's output for the systemd one-liner).
#
# Usage (from a fresh Ubuntu PC, no repo yet):
#
#   # Option 1 — download and run directly (recommended):
#   curl -fsSL https://raw.githubusercontent.com/waledroid/IsiDetector/dev/install.sh | bash
#
#   # Option 2 — download first, review, then run:
#   curl -fsSLo install.sh https://raw.githubusercontent.com/waledroid/IsiDetector/dev/install.sh
#   chmod +x install.sh
#   ./install.sh
#
#   # Option 3 — custom install dir / branch:
#   INSTALL_DIR="$HOME/isidetector" BRANCH="main" ./install.sh
#
# ============================================================================

set -euo pipefail

# ── Configuration (overridable via environment variables) ───────────────────
REPO_URL="${REPO_URL:-https://github.com/waledroid/IsiDetector.git}"
BRANCH="${BRANCH:-dev}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/logistic}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-ask}"   # yes | no | ask

# ── Colors ──────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
    RED='\033[0;31m';   BOLD='\033[1m';     NC='\033[0m'
else
    GREEN=''; YELLOW=''; CYAN=''; RED=''; BOLD=''; NC=''
fi

info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[  OK]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $1" >&2; exit 1; }
header()  {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo ""
}

# ── Stage 1: Preflight checks ───────────────────────────────────────────────
header "Stage 1/4 — Preflight checks"

# Must not be root (Docker group setup later needs a regular user)
if [ "$(id -u)" -eq 0 ]; then
    fail "Do not run this script as root. Run as your regular user — sudo will be called where needed."
fi

# Must have sudo available
if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo is required but not installed."
fi

# Verify sudo works (cached credential or prompt)
if ! sudo -n true 2>/dev/null; then
    info "This script needs sudo for package installs. You may be prompted for your password."
    sudo -v || fail "sudo authentication failed."
fi

# Must be Ubuntu (22.04 or 24.04 ideally; warn on others)
if [ -r /etc/os-release ]; then
    . /etc/os-release
    case "$ID" in
        ubuntu)
            case "$VERSION_ID" in
                22.04|24.04) success "Ubuntu ${VERSION_ID} (${VERSION_CODENAME}) — supported" ;;
                *)           warn   "Ubuntu ${VERSION_ID} — run_start.sh tested on 22.04 / 24.04 only. Proceeding." ;;
            esac
            ;;
        debian)   warn "Debian detected — Ubuntu scripts often work. Proceeding." ;;
        *)        warn "Distro '${ID}' not tested — proceeding anyway." ;;
    esac
else
    warn "Could not read /etc/os-release. Proceeding."
fi

# Warn if running in WSL2 — bare-metal Linux is strongly preferred for the
# production sort-trigger use case (see docs/deployment.md and the UDP
# latency analysis for why).
if grep -qi microsoft /proc/version 2>/dev/null; then
    warn "WSL2 detected. For a production sorter PC, prefer native Ubuntu."
    warn "If this PC will be shipped to a customer, install Ubuntu to bare metal."
fi

# ── Stage 2: Install git + curl if missing ──────────────────────────────────
header "Stage 2/4 — Base dependencies (git + curl)"

APT_UPDATED=false
ensure_pkg() {
    if ! command -v "$1" >/dev/null 2>&1; then
        if [ "$APT_UPDATED" = false ]; then
            info "Refreshing apt package index..."
            sudo apt-get update -qq
            APT_UPDATED=true
        fi
        info "Installing '$2'..."
        sudo apt-get install -y -qq "$2" >/dev/null
        success "$2 installed"
    else
        success "$1 already present"
    fi
}

ensure_pkg git git
ensure_pkg curl curl

# ── Stage 3: Clone the repo ─────────────────────────────────────────────────
header "Stage 3/4 — Clone repository"

info "Source:  ${REPO_URL}"
info "Branch:  ${BRANCH}"
info "Target:  ${INSTALL_DIR}"
echo ""

if [ -d "$INSTALL_DIR" ]; then
    if [ -d "$INSTALL_DIR/.git" ]; then
        info "Existing clone found at ${INSTALL_DIR} — updating"
        cd "$INSTALL_DIR"
        git fetch origin "${BRANCH}" --quiet
        git checkout "${BRANCH}" --quiet
        git pull --ff-only origin "${BRANCH}"
        success "Repository updated to latest ${BRANCH}"
    else
        fail "${INSTALL_DIR} exists and is not a git clone. Move it aside or pick a different INSTALL_DIR."
    fi
else
    info "Cloning (shallow, dev branch only)..."
    git clone --branch "${BRANCH}" --depth 1 "${REPO_URL}" "${INSTALL_DIR}"
    cd "$INSTALL_DIR"
    success "Cloned to ${INSTALL_DIR}"
fi

# Make runtime scripts executable
chmod +x run_start.sh up.sh compress.sh install.sh 2>/dev/null || true
success "Scripts made executable"

# ── Stage 4: Optionally bootstrap Docker ────────────────────────────────────
header "Stage 4/4 — Bootstrap (Docker install + image build)"

case "$RUN_BOOTSTRAP" in
    yes) do_bootstrap=true  ;;
    no)  do_bootstrap=false ;;
    ask)
        if [ -t 0 ]; then
            read -rp "Run ./run_start.sh now to install Docker and build the image? [Y/n] " ans
            case "$ans" in
                n|N|no|NO) do_bootstrap=false ;;
                *)         do_bootstrap=true  ;;
            esac
        else
            info "Non-interactive shell — skipping bootstrap. Run ./run_start.sh manually."
            do_bootstrap=false
        fi
        ;;
esac

if [ "$do_bootstrap" = true ]; then
    info "Handing off to ./run_start.sh ..."
    echo ""
    ./run_start.sh
else
    info "Skipped. Run it later with:  cd ${INSTALL_DIR} && ./run_start.sh"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
header "Install complete"

echo -e "  ${BOLD}Repo:${NC}       ${INSTALL_DIR} (${BRANCH})"
echo -e ""
echo -e "  ${BOLD}Next steps:${NC}"
echo -e "    1. ${CYAN}cd ${INSTALL_DIR}${NC}"
if [ "$do_bootstrap" = false ]; then
    echo -e "    2. ${CYAN}./run_start.sh${NC}   — install Docker, build image (one-time, ~5-10 min)"
    echo -e "    3. Copy your trained model weights into:"
else
    echo -e "    2. Copy your trained model weights into:"
fi
echo -e "         ${CYAN}${INSTALL_DIR}/models/yolo/<date>/openvino/${NC}   (.xml + .bin)"
echo -e "         or"
echo -e "         ${CYAN}${INSTALL_DIR}/runs/segment/models/yolo/<date>/weights/${NC}"
echo -e "       From your dev machine, for example:"
echo -e "         ${CYAN}scp -r models/yolo/<date>/openvino USER@\$(hostname -I|awk '{print\$1}'):${INSTALL_DIR}/models/yolo/<date>/${NC}"
if [ "$do_bootstrap" = false ]; then
    echo -e "    4. ${CYAN}./up.sh${NC}          — start the stack and open the dashboard"
else
    echo -e "    3. ${CYAN}./up.sh${NC}          — start the stack and open the dashboard"
fi
echo -e ""
echo -e "  ${BOLD}Optional — run on every boot (production):${NC}"
echo -e "    ${CYAN}sudo crontab -e${NC}"
echo -e "    Add:  ${CYAN}@reboot cd ${INSTALL_DIR} && ./up.sh${NC}"
echo -e ""
echo -e "  ${BOLD}Dashboard will be at:${NC} ${CYAN}http://localhost:9501${NC}"
echo -e "  ${BOLD}UDP sort-trigger feed:${NC}  ${CYAN}127.0.0.1:9502${NC}"
echo -e ""
