#!/bin/bash
# ============================================================================
# IsiDetector — Docker Install & Launch Script
# Works on any fresh Ubuntu 22.04 / 24.04 machine (with or without GPU)
# Usage: chmod +x run_start.sh && ./run_start.sh
# ============================================================================

set -e

# ── Colors & Helpers ────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[  OK]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[SKIP]${NC}  $1"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $1"; }
header()  {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo ""
}

# Project directory = where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get Ubuntu codename once (needed for Docker repo)
CODENAME=$(. /etc/os-release 2>/dev/null && echo "$VERSION_CODENAME" || echo "noble")

# ── Stage 1: Detect Hardware & Platform ─────────────────────────────────────
header "Stage 1/7 — Hardware & Platform Detection"

# Detect WSL
IS_WSL=false
HAS_X11=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
    success "Platform: WSL2 (Windows Subsystem for Linux)"
    # Check for WSLg (GUI support)
    if [ -d "/mnt/wslg" ] && [ -n "$DISPLAY" ]; then
        HAS_X11=true
        success "WSLg detected (GUI support via X11/Wayland)"
    else
        warn "WSLg not available — cv2.imshow() will not work"
        info "Install Windows 11 22H2+ for WSLg support"
    fi
else
    success "Platform: Native Linux"
    if [ -n "$DISPLAY" ]; then
        HAS_X11=true
        success "X11 display available (DISPLAY=$DISPLAY)"
    else
        warn "No DISPLAY set — cv2.imshow() will not work (headless server)"
    fi
fi

# Detect GPU
HAS_GPU=false
GPU_NAME=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
    success "NVIDIA GPU detected: ${GPU_NAME}"
    info "Mode: GPU inference (ONNX CUDA + TensorRT + native PyTorch)"
else
    warn "No NVIDIA GPU detected"
    info "Mode: CPU inference (OpenVINO optimized)"
fi

# ── Stage 2: Install Docker Engine ──────────────────────────────────────────
header "Stage 2/7 — Docker Engine"

# Check if Docker Engine (not Docker Desktop) is actually working
DOCKER_OK=false
if dpkg -l docker-ce &>/dev/null 2>&1; then
    DOCKER_VER=$(dpkg -l docker-ce 2>/dev/null | grep '^ii' | awk '{print $3}' | head -1)
    if [ -n "$DOCKER_VER" ]; then
        success "Docker Engine already installed (${DOCKER_VER})"
        DOCKER_OK=true
    fi
fi

if [ "$DOCKER_OK" = false ]; then
    info "Installing Docker Engine..."

    # Prerequisites
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg >/dev/null 2>&1
    success "Prerequisites installed"

    # Docker GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
        success "Docker GPG key added"
    else
        warn "Docker GPG key already exists"
    fi

    # Docker repository
    sudo bash -c "echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable' > /etc/apt/sources.list.d/docker.list"
    success "Docker repository configured (Ubuntu ${CODENAME})"

    # Install Docker packages
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-buildx-plugin >/dev/null 2>&1
    success "Docker Engine installed"
fi

# ── Stage 3: Docker Service & Permissions ───────────────────────────────────
header "Stage 3/7 — Docker Service & Permissions"

# Start Docker daemon
if sudo service docker status &>/dev/null 2>&1; then
    success "Docker daemon is running"
else
    info "Starting Docker daemon..."
    sudo service docker start
    sleep 2
    if sudo service docker status &>/dev/null 2>&1; then
        success "Docker daemon started"
    else
        fail "Could not start Docker daemon"
        info "Try: sudo dockerd &"
        exit 1
    fi
fi

# Add user to docker group
if getent group docker &>/dev/null; then
    if groups "$USER" | grep -qw docker; then
        success "User '${USER}' is in docker group"
    else
        info "Adding '${USER}' to docker group..."
        sudo usermod -aG docker "$USER"
        success "Added to docker group"
        warn "You may need to run: newgrp docker (or log out and back in)"
    fi
else
    fail "Docker group does not exist — Docker may not have installed correctly"
    exit 1
fi

# Verify Docker works and determine if sudo is needed
SUDO_DOCKER=""
if docker info &>/dev/null 2>&1; then
    DOCKER_VER=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
    success "Docker Engine v${DOCKER_VER} — working"
elif sudo docker info &>/dev/null 2>&1; then
    SUDO_DOCKER="sudo"
    DOCKER_VER=$(sudo docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
    success "Docker Engine v${DOCKER_VER} — working (via sudo)"
    info "Tip: run 'newgrp docker' in a new terminal to avoid sudo next time"
else
    fail "Docker is not responding"
    exit 1
fi

# ── Stage 4: NVIDIA Container Toolkit (GPU only) ───────────────────────────
header "Stage 4/7 — Runtime Setup"

if [ "$HAS_GPU" = true ]; then
    if dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
        success "NVIDIA Container Toolkit already installed"
    else
        info "Installing NVIDIA Container Toolkit..."

        # Add NVIDIA GPG key
        if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        fi

        # Add NVIDIA repository
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

        sudo apt-get update -qq
        sudo apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1
        sudo nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1
        sudo service docker restart
        sleep 2

        success "NVIDIA Container Toolkit installed and configured"
    fi
else
    warn "No GPU — skipping NVIDIA Container Toolkit"
    info "OpenVINO (.xml) models recommended for best CPU performance"
fi

# ── Stage 5: Verify Runtime ─────────────────────────────────────────────────
header "Stage 5/7 — Runtime Verification"

if [ "$HAS_GPU" = true ]; then
    info "Testing GPU access inside Docker..."
    if $SUDO_DOCKER docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
        success "GPU accessible inside Docker containers"
    else
        warn "GPU not accessible inside Docker — falling back to CPU mode"
        info "Check NVIDIA driver version and container toolkit installation"
        HAS_GPU=false
    fi
else
    info "Testing Docker engine (CPU mode)..."
    if $SUDO_DOCKER docker run --rm hello-world &>/dev/null 2>&1; then
        success "Docker engine working (CPU mode)"
    else
        fail "Docker engine test failed"
        exit 1
    fi
fi

# ── Stage 6: Build & Launch IsiDetector ─────────────────────────────────────
header "Stage 6/7 — Build & Launch IsiDetector"

info "Project directory: ${SCRIPT_DIR}"

# Create runtime directories
mkdir -p models configs isitec_app/logs isitec_app/uploads
success "Runtime directories ready"

# Determine compose command (GPU vs CPU, with sudo if needed)
if [ "$HAS_GPU" = true ]; then
    COMPOSE_CMD="$SUDO_DOCKER docker compose"
    info "Using GPU compose profile"
else
    COMPOSE_CMD="$SUDO_DOCKER docker compose -f docker-compose.yml -f docker-compose.cpu.yml"
    info "Using CPU compose profile (no GPU reservation)"
fi

# Stop existing container if running (preserves it, just stops)
if $COMPOSE_CMD ps -q 2>/dev/null | grep -q .; then
    info "Stopping existing container..."
    $COMPOSE_CMD stop 2>/dev/null
    success "Previous container stopped"
else
    warn "No existing container running"
fi

# Build the image (cached layers are reused automatically)
info "Building Docker image (first build takes 5-10 minutes, subsequent builds are cached)..."
$COMPOSE_CMD build
success "Docker image ready"

# ── Stage 7: X11 / GUI Check ────────────────────────────────────────────────
header "Stage 7/7 — GUI Display Check"

if [ "$HAS_X11" = true ]; then
    success "X11 forwarding enabled — cv2.imshow() will work inside container"
    info "Run: docker compose exec web python scripts/run_live.py --weights <model> --source <stream>"
else
    warn "No GUI display — run_live.py (OpenCV window) will not work"
    info "Use the Flask web app at http://localhost:9501 instead"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
header "IsiDetector Ready"

echo -e "  ${BOLD}Platform:${NC}     $([ "$IS_WSL" = true ] && echo "WSL2" || echo "Native Linux") $([ "$HAS_X11" = true ] && echo "(GUI ready)" || echo "(headless)")"
echo -e "  ${BOLD}Hardware:${NC}     $([ "$HAS_GPU" = true ] && echo "GPU (${GPU_NAME})" || echo "CPU-only (OpenVINO)")"
echo -e "  ${BOLD}Dashboard:${NC}    http://localhost:9501"
echo -e "  ${BOLD}Docs:${NC}         http://localhost:9501/docs"
echo -e "  ${BOLD}UDP Target:${NC}   127.0.0.1:9502"
echo -e ""
echo -e "  ${BOLD}Model weights:${NC} Place .pt / .onnx / .xml files in ${SCRIPT_DIR}/models/"
echo -e "  ${BOLD}Dev access:${NC}   Double-click logo, password: Isitec69+"
echo -e ""

if [ "$HAS_GPU" = false ]; then
    echo -e "  ${YELLOW}Tip:${NC} Use OpenVINO (.xml) models for best CPU performance."
    echo -e "  ${YELLOW}Generate:${NC} python -m src.inference.export_engine --model-dir <dir> --format openvino"
    echo -e ""
fi

echo -e "  ${BOLD}Commands:${NC}"
echo -e "    Start:      ${CYAN}docker compose up -d${NC}"
echo -e "    Stop:       ${CYAN}docker compose stop${NC}"
echo -e "    Logs:       ${CYAN}docker compose logs -f${NC}"
echo -e "    Shell:      ${CYAN}docker compose exec web bash${NC}"
echo -e "    Live feed:  ${CYAN}docker compose exec web python scripts/run_live.py --weights <model> --source <stream>${NC}"
echo -e "    Rebuild:    ${CYAN}docker compose build${NC}"
echo -e ""
echo -e "  ${BOLD}Launching...${NC}"
echo -e ""

# Launch in background, then follow logs
$COMPOSE_CMD up -d
success "Container started"
echo ""
info "Following logs (Ctrl+C to stop watching — container keeps running)"
echo ""
$COMPOSE_CMD logs -f
