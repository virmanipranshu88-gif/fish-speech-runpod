# Fish Speech TTS RunPod Serverless Handler
# Based on official Fish Speech Docker image

FROM fishaudio/fish-speech:latest

# Switch to root to install packages
USER root

WORKDIR /app

# Install runpod into the venv using uv
RUN uv pip install --python /app/.venv/bin/python runpod>=1.6.0

# Copy handler
COPY handler.py .

# Environment variables
ENV CHECKPOINT_PATH=/runpod-volume/fish-speech/checkpoints/openaudio-s1-mini
ENV DECODER_CHECKPOINT=/runpod-volume/fish-speech/checkpoints/openaudio-s1-mini/codec.pth
ENV DECODER_CONFIG=modded_dac_vq
ENV PYTHONUNBUFFERED=1

# Override the base image's ENTRYPOINT (which runs run_webui.py)
ENTRYPOINT []

# Run handler using the venv Python (where torch is installed)
CMD ["/app/.venv/bin/python", "-u", "handler.py"]
