FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

USER root
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    git curl wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /app/.venv

RUN /app/.venv/bin/pip install --upgrade pip uv

RUN /app/.venv/bin/uv pip install \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu126

RUN /app/.venv/bin/uv pip install \
    "git+https://github.com/fishaudio/fish-speech.git@main" \
    "runpod>=1.6.0"

COPY handler.py .

ENV PATH="/app/.venv/bin:$PATH"
ENV LLAMA_CHECKPOINT_PATH=/runpod-volume/fish-speech/checkpoints/s2-pro
ENV DECODER_CHECKPOINT_PATH=/runpod-volume/fish-speech/checkpoints/s2-pro/codec.pth
ENV DECODER_CONFIG_NAME=modded_dac_vq
ENV VOICE_HINDI=/runpod-volume/fish-audio/voices/hindi/reference_hindi.wav
ENV VOICE_ENGLISH=/runpod-volume/fish-audio/voices/english/reference_english.wav
ENV COMPILE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONUNBUFFERED=1

ENTRYPOINT []
CMD ["/app/.venv/bin/python", "-u", "handler.py"]