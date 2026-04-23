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
COPY deploy/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# ── Application code ────────────────────────────────────────────────────────
# After the bucket restructure, the ML core lives under isidet/ and the web
# app under webapp/. Copy only inference + web-app code so the image stays
# small; training code, runs/, models/ artefacts come in via volumes.
COPY isidet/src/inference/  isidet/src/inference/
COPY isidet/src/shared/     isidet/src/shared/
COPY isidet/src/utils/      isidet/src/utils/
COPY isidet/src/preprocess/ isidet/src/preprocess/
COPY isidet/src/__init__.py isidet/src/__init__.py
COPY isidet/configs/        isidet/configs/
COPY webapp/isitec_app/     webapp/isitec_app/

# ── MkDocs (optional — pre-built site mounted via compose) ──────────────────
# To build docs: cd mkdocs && mkdocs build
# Docs served at /docs if webapp/isitec_app/static/docs/ exists
RUN mkdir -p webapp/isitec_app/static/docs

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
# PYTHONPATH points at isidet/ so every `from src.X` in the codebase keeps
# resolving. The bucket rename never touched any of the ~55 `from src.`
# imports — that's the whole point of putting isidet/ on the path.
ENV PYTHONPATH=/opt/isitec/isidet
ENV PORT=9501
ENV DEV_PASSWORD=change-me
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;10.0;12.0;12.0+PTX"
ENV TZ=Europe/Paris

# ── Runtime directories (will be volume-mounted) ────────────────────────────
RUN mkdir -p isidet/models isidet/runs isidet/logs webapp/isitec_app/logs webapp/isitec_app/uploads

EXPOSE 9501
EXPOSE 9502/udp

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# CWD stays /opt/isitec so the web app's `Path("isidet/...")` resolutions
# and volume-mounted paths all line up.
CMD ["python", "webapp/isitec_app/app.py"]
