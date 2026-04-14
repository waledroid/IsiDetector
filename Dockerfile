# ============================================================================
# IsiDetector — Production Inference Container
# Build:  docker build -t isitec-visionai .
# Run:    docker run --gpus all -p 9501:9501 -p 9502:9502/udp \
#           -v ./models:/opt/isitec/models \
#           -v ./configs:/opt/isitec/configs \
#           isitec-visionai
# ============================================================================

# ── CUDA runtime + inference app ─────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (OpenCV, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    curl tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /opt/isitec

# ── Python dependencies ─────────────────────────────────────────────────────
# Install PyTorch + torchvision with CUDA 12.8 support
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install ONNX Runtime GPU (uninstall CPU version first to avoid conflict)
RUN pip install --no-cache-dir onnxruntime-gpu

# Install remaining deploy dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# ── Application code ────────────────────────────────────────────────────────
# Only inference + web app code (no training code, no data)
COPY src/inference/  src/inference/
COPY src/shared/     src/shared/
COPY src/utils/      src/utils/
COPY src/preprocess/ src/preprocess/
COPY src/__init__.py src/__init__.py
COPY isitec_app/     isitec_app/
COPY configs/        configs/
COPY scripts/        scripts/

# ── MkDocs (optional — mount pre-built site or run mkdocs build manually) ───
# To build docs: mkdocs build && docker compose restart
# Docs served at /docs if isitec_app/static/docs/ exists
RUN mkdir -p isitec_app/static/docs

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/isitec
ENV PORT=9501
ENV DEV_PASSWORD=Isitec69+
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;10.0;12.0;12.0+PTX"
ENV TZ=Europe/Paris

# ── Runtime directories (will be volume-mounted) ────────────────────────────
RUN mkdir -p models isitec_app/logs isitec_app/uploads

EXPOSE 9501
EXPOSE 9502/udp

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

CMD ["python", "isitec_app/app.py"]
