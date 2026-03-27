FROM python:3.11-slim

WORKDIR /app

# Dependencies in a separate layer so they're cached across rebuilds
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App
COPY app.py .

# Model baked into the image — CI runs export.py first, then docker build
# model/unet_int8.onnx  ~94 MB
# model/config.yaml
COPY model/ model/

ENV MODEL_PATH=model/unet_int8.onnx \
    MODEL_CONFIG=model/config.yaml

EXPOSE 7860

# PORT env var is set automatically by Cloud Run, HF Spaces, Fly.io etc.
# Defaults to 7860 (HF Spaces) if not set.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
