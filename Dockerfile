# Reproducible CPU environment for safe-drive (training, eval, baselines, tests).
# The workload is CPU-bound (MetaDrive physics on CPU; the policy net is tiny),
# so a CPU image reproduces results faithfully. For GPU training, swap the base
# for an nvidia/cuda runtime and install the matching torch wheel.
FROM python:3.10-slim

# System libraries MetaDrive / Panda3D need, even headless (no display).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libxrender1 \
        libsm6 \
        libxext6 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest imageio

# Project code.
COPY . .

ENV PYTHONUNBUFFERED=1

# Default: run the test suite (proves the image is wired up correctly).
# Override to train/eval, e.g.:
#   docker run --rm safe-drive python eval/baselines.py --policy random
#   docker run --rm safe-drive python train.py --test
CMD ["python", "-m", "pytest", "tests/", "-q"]