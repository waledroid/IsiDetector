# Standalone / dev-mode Flask Dockerfile. Most deployments use the
# canonical deploy/Dockerfile instead — this file is kept for one-off
# service-level builds.
#
# Build context must be the repo root (~/logistic).
# Command: docker build -t isitec-visionai -f webapp/isitec_app/Dockerfile .

FROM python:3.10-slim

WORKDIR /opt/isitec

# Install necessary system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY webapp/isitec_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code — new bucket layout
COPY isidet/src/   /opt/isitec/isidet/src/
COPY isidet/configs/ /opt/isitec/isidet/configs/
COPY webapp/isitec_app/ /opt/isitec/webapp/isitec_app/

# Environment configurations. PYTHONPATH at isidet/ keeps `from src.X`
# resolving without any import rewrites.
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/isitec/isidet

EXPOSE 9501
EXPOSE 9502

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:9501/ || exit 1

# Run Flask Application
CMD ["python", "webapp/isitec_app/app.py"]
