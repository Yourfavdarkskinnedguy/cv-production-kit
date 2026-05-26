# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — cv-production-kit
#
# Builds a GPU-enabled container for running the CV pipeline.
#
# Build:
#   docker build -t cv-production-kit .
#
# Run on webcam (needs --device for USB camera):
#   docker run --gpus all --device /dev/video0 -it cv-production-kit
#
# Run on RTSP stream (no device needed):
#   docker run --gpus all -e CV_SOURCE="rtsp://192.168.1.10:554/stream" \
#              -it cv-production-kit
#
# Run headless (no display window):
#   docker run --gpus all -e CV_SOURCE="data/test.mp4" \
#              -v $(pwd)/data:/app/data \
#              -it cv-production-kit python scripts/run_pipeline.py --no-display
# ─────────────────────────────────────────────────────────────────────────────

# ── Base image ────────────────────────────────────────────────────────────────
# Use NVIDIA CUDA image so TensorRT and ONNX Runtime GPU work out of the box.
# Match the CUDA version to your host driver.
# Check your driver version: nvidia-smi | grep "CUDA Version"
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── Build arguments ───────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-setuptools \
    # OpenCV runtime dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Video device support
    v4l-utils \
    # Useful for debugging in the container
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python symlink ────────────────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first — Docker caches this layer.
# Only invalidated when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir onnxruntime-gpu>=1.16.0 \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# ── Create directories the pipeline expects ───────────────────────────────────
RUN mkdir -p models logs data

# ── Environment variables ─────────────────────────────────────────────────────
# CV_SOURCE overrides camera.source in the config at runtime.
# Set to a video file path, RTSP URL, or integer webcam index.
ENV CV_SOURCE=""
ENV CV_CONFIG="configs/default.yaml"
ENV PYTHONUNBUFFERED=1

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from cv_kit import Pipeline; print('ok')" || exit 1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "scripts/run_pipeline.py", "--no-display"]