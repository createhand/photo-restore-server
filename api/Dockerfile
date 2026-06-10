FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r /app/requirements.txt
RUN python - <<'PY'
from pathlib import Path

import torchvision.transforms

shim_path = Path(torchvision.transforms.__file__).parent / "functional_tensor.py"
if not shim_path.exists():
    shim_path.write_text(
        "from torchvision.transforms.functional import rgb_to_grayscale\n",
        encoding="utf-8",
    )
PY

COPY . /app

RUN mkdir -p /app/models /app/input /app/output

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
