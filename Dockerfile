FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
USER root
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    git curl wget ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create venv
RUN uv venv /app/.venv --python 3.12
ENV PATH="/app/.venv/bin:$PATH"

# Install PyTorch with CUDA 12.6
RUN uv pip install \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu126

# Clone fish-speech main (S2 code) and install
RUN git clone https://github.com/fishaudio/fish-speech.git /app/fish-speech
RUN cd /app/fish-speech && uv pip install -e .

# Install runpod
RUN uv pip install "runpod>=1.6.0"

COPY handler.py .

ENV PYTHONPATH=/app/fish-speech
ENV LLAMA_CHECKPOINT_PATH=/runpod-volume/fish-speech/checkpoints/s2-pro
ENV DECODER_CHECKPOINT_PATH=/runpod-volume/fish-speech/checkpoints/s2-pro/codec.pth
ENV DECODER_CONFIG_NAME=modded_dac_vq
ENV VOICE_HINDI=/runpod-volume/fish-audio/voices/hindi/reference_hindi.wav
ENV VOICE_ENGLISH=/runpod-volume/fish-audio/voices/english/reference_english.wav
ENV COMPILE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONUNBUFFERED=1

ENTRYPOINT []
CMD ["python", "-u", "handler.py"]