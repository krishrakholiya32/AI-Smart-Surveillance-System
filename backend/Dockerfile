FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
# The dedicated CPU-only wheel index only publishes x86_64 builds — on
# arm64 (e.g. Oracle Cloud Ampere), fall back to plain PyPI, which ships
# CPU-only wheels for that platform by default (no CUDA variant exists).
RUN ARCH=$(dpkg --print-architecture) \
 && if [ "$ARCH" = "amd64" ]; then \
        pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1; \
    fi \
 && pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY scripts/export_onnx.py ./scripts/export_onnx.py
RUN chmod +x entrypoint.sh

# Models are mounted as a volume; storage dir
RUN mkdir -p /app/storage/alerts /app/models

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
