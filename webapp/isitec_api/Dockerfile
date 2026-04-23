# Standalone / dev-mode FastAPI Dockerfile. Most deployments use the
# canonical deploy/Dockerfile instead — this file is kept for one-off
# service-level builds.
#
# Build context must be the repo root (~/logistic).
# Command: docker build -t isitec-visionai-api -f webapp/isitec_api/Dockerfile .

FROM python:3.10-slim

WORKDIR /opt/isitec

# Install necessary system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY webapp/isitec_api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code — new bucket layout
COPY isidet/src/   /opt/isitec/isidet/src/
COPY isidet/configs/ /opt/isitec/isidet/configs/
COPY webapp/isitec_api/ /opt/isitec/webapp/isitec_api/

# Environment configurations. PYTHONPATH at isidet/ keeps `from src.X`
# resolving; webapp/ must be on sys.path too (done inside app.py).
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/isitec/isidet:/opt/isitec/webapp

EXPOSE 9501
EXPOSE 9502

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:9501/ || exit 1

# Run FastAPI Application
CMD ["python", "-m", "uvicorn", "isitec_api.app:app", "--host", "0.0.0.0", "--port", "9501"]
