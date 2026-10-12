# Image GPU (NVIDIA). Pour un usage CPU seulement, remplacez la base par python:3.12-slim.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-qwen.txt ./
ARG WITH_QWEN=0
RUN pip3 install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_QWEN" = "1" ]; then pip3 install --no-cache-dir -r requirements-qwen.txt; fi
COPY transcriber ./transcriber

ENV TRANSCRIBER_DATA_DIR=/data HF_HOME=/models
VOLUME ["/data", "/models"]
EXPOSE 8000
CMD ["python3", "-m", "transcriber", "--host", "0.0.0.0", "--port", "8000"]
