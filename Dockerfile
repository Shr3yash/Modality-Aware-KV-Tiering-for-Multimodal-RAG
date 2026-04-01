# =============================================================================
# Stage: dev (CPU-only, no vLLM)
# Use this to run tests and develop Phase 1 cache code without a GPU.
#   docker build --target dev -t kv-tiering:dev .
#   docker run --rm kv-tiering:dev pytest tests/
# =============================================================================
FROM python:3.10-slim AS dev

WORKDIR /app

# System deps needed by torch/Pillow/faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install all deps EXCEPT vllm (not integrated yet, Linux GPU only)
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

CMD ["pytest", "tests/", "-v"]


# =============================================================================
# Stage: gpu (CUDA 12.1 + vLLM, for Phase 2+ serving)
# Requires nvidia-container-toolkit on the host.
#   docker build --target gpu -t kv-tiering:gpu .
#   docker run --gpus all --rm kv-tiering:gpu
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.10 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip

# Install PyTorch first (CUDA 12.1 wheel) before the rest of the deps
RUN pip install --no-cache-dir \
    torch==2.2.1+cu121 \
    torchvision==0.17.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir \
    $(grep -v '^\s*torch\|^\s*torchvision' requirements.txt) \
    pytest==8.0.0 pytest-asyncio==0.23.0

COPY . .

# SSD cache directory
RUN mkdir -p /tmp/kv_ssd_cache

CMD ["python", "-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
