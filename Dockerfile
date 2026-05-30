FROM fishaudio/fish-speech:latest-server-cuda

USER root
WORKDIR /app

RUN uv pip install --python /app/.venv/bin/python "runpod>=1.6.0"

COPY handler.py .

ENV LLAMA_CHECKPOINT_PATH=/runpod-volume/fish-speech/checkpoints/openaudio-s1-mini
ENV DECODER_CHECKPOINT_PATH=/runpod-volume/fish-speech/checkpoints/openaudio-s1-mini/codec.pth
ENV DECODER_CONFIG_NAME=modded_dac_vq
ENV VOICE_HINDI=/runpod-volume/fish-audio/voices/hindi/reference_hindi.wav
ENV VOICE_ENGLISH=/runpod-volume/fish-audio/voices/english/reference_english.wav
ENV PYTHONUNBUFFERED=1
ENV COMPILE=0

ENTRYPOINT []
CMD ["/app/.venv/bin/python", "-u", "handler.py"]