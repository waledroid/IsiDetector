# Build context must be ~/logistic
# Command: docker build -t isitec-visionai -f isitec_app/Dockerfile .

FROM python:3.10-slim

WORKDIR /opt/isitec

# Install necessary system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY isitec_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /opt/isitec/src/
COPY isitec_app/ /opt/isitec/isitec_app/

# Environment configurations
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/isitec

EXPOSE 9501

# Run Flask Application
CMD ["python", "isitec_app/app.py"]
