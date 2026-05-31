FROM fishaudio/fish-speech:latest

USER root
WORKDIR /app

RUN uv pip install --python /app/.venv/bin/python "runpod>=1.6.0"

COPY handler.py .

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