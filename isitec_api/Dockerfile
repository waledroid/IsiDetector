# Build context must be ~/logistic
# Command: docker build -t isitec-visionai-api -f isitec_api/Dockerfile .

FROM python:3.10-slim

WORKDIR /opt/isitec

# Install necessary system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY isitec_api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /opt/isitec/src/
COPY isitec_api/ /opt/isitec/isitec_api/

# Environment configurations
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/isitec

EXPOSE 9501
EXPOSE 9502

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:9501/ || exit 1

# Run FastAPI Application
CMD ["python", "-m", "uvicorn", "isitec_api.app:app", "--host", "0.0.0.0", "--port", "9501"]
