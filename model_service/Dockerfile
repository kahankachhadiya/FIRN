# Base Image: NVIDIA CUDA 12.4 + cuDNN 9 (required by onnxruntime-gpu CUDAExecutionProvider)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* || \
    (apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*)

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

WORKDIR /app

# Set environment variables
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HUB_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,expandable_segments:True
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Copy and install requirements.txt
COPY model_service/requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.4 support first
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements (exclude torch to avoid overwriting CUDA build)
RUN grep -v "^torch==" /app/requirements.txt > /tmp/requirements_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

# Copy pre-downloaded models from local directory
# Run model_service/download_models_local.py first to download models
COPY model_service/models /app/models

# Copy entry points
COPY model_service/api_server.py /app/api_server.py
COPY model_service/gpu_worker.py /app/gpu_worker.py

# Expose port 7997
EXPOSE 7997

# Launch GPU worker and API server together
CMD ["bash", "-c", "python3 /app/gpu_worker.py & uvicorn api_server:app --host 0.0.0.0 --port 7997 --workers 1"]
