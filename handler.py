"""
Fish Speech TTS RunPod Serverless Handler

Provides voice cloning TTS using Fish Speech OpenAudio S1-mini model.
Uses Pranshu's cloned Hindi/English voice from network volume.

Input format:
{
    "text": "The text to synthesize",
    "language": "hindi" | "english",          # selects voice reference
    "reference_audio_base64": "...",           # optional override
    "emotion_marker": "(sincere) (soft tone)", # optional
    "temperature": 0.9,
    "top_p": 0.85,
    "repetition_penalty": 1.1
}

Output format:
{
    "audio_base64": "base64 encoded WAV audio",
    "sample_rate": 44100
}
"""

import runpod
import base64
import io
import os
import tempfile
import traceback

import numpy as np
import torch
import torchaudio

# ── Config — uses correct env var names from official Fish Audio docs ─────────
CHECKPOINT_PATH    = os.environ.get("LLAMA_CHECKPOINT_PATH",    "/app/checkpoints/openaudio-s1-mini")
DECODER_CHECKPOINT = os.environ.get("DECODER_CHECKPOINT_PATH",  f"{CHECKPOINT_PATH}/codec.pth")
DECODER_CONFIG     = os.environ.get("DECODER_CONFIG_NAME",      "modded_dac_vq")
VOICE_HINDI        = os.environ.get("VOICE_HINDI",  "/runpod-volume/fish-audio/voices/hindi/reference_hindi.wav")
VOICE_ENGLISH      = os.environ.get("VOICE_ENGLISH", "/runpod-volume/fish-audio/voices/english/reference_english.wav")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
HALF_PRECISION     = torch.cuda.is_available()
COMPILE_MODEL      = os.environ.get("COMPILE", "0") == "1"

MIN_TEXT_LENGTH      = 1
MAX_TEXT_LENGTH      = 2000
MIN_REFERENCE_DURATION = 3.0
OUTPUT_SAMPLE_RATE   = 44100
AMPLITUDE            = 32768
NARRATION_STYLE      = "(sincere) (soft tone)"

tts_engine = None


def verify_model_present():
    """Verify model weights are present on network volume."""
    errors = []
    if not os.path.exists(CHECKPOINT_PATH) or not os.listdir(CHECKPOINT_PATH):
        errors.append(f"Checkpoint missing: {CHECKPOINT_PATH}")
    if not os.path.exists(DECODER_CHECKPOINT):
        errors.append(f"Decoder missing: {DECODER_CHECKPOINT}")
    if not os.path.exists(VOICE_HINDI):
        errors.append(f"Hindi voice missing: {VOICE_HINDI}")
    if not os.path.exists(VOICE_ENGLISH):
        errors.append(f"English voice missing: {VOICE_ENGLISH}")
    if errors:
        raise RuntimeError("Volume errors:\n" + "\n".join(errors))
    print(f"✅ Volume verified")
    print(f"   Checkpoint : {CHECKPOINT_PATH}")
    print(f"   Decoder    : {DECODER_CHECKPOINT}")
    print(f"   Hindi voice: {VOICE_HINDI}")
    print(f"   English    : {VOICE_ENGLISH}")
    print(f"   Device     : {DEVICE}")


verify_model_present()

# Fish Speech imports — uses vqgan decoder (correct for server-cuda image)
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


def load_models():
    """Load Fish Speech models on cold start."""
    global tts_engine

    print(f"Loading Fish Speech models from {CHECKPOINT_PATH}...")
    print(f"Device: {DEVICE}, Half precision: {HALF_PRECISION}, Compile: {COMPILE_MODEL}")

    precision = torch.float16 if HALF_PRECISION else torch.bfloat16

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        precision=precision,
        compile=COMPILE_MODEL,
    )
    print("LLAMA model loaded")

    decoder_model = load_decoder_model(
        config_name=DECODER_CONFIG,
        checkpoint_path=DECODER_CHECKPOINT,
        device=DEVICE,
    )
    print(f"Decoder model loaded, sample rate: {decoder_model.sample_rate}")

    tts_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=COMPILE_MODEL,
    )
    print("TTS inference engine created")

    # Warmup
    print("Warming up models...")
    try:
        warmup_request = ServeTTSRequest(
            text="Hello, this is a warmup test.",
            references=[],
            temperature=0.7,
            repetition_penalty=1.2,
            format="wav",
        )
        for result in tts_engine.inference(warmup_request):
            pass
        print("Models warmed up successfully")
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

    print("Fish Speech models ready!")


def get_voice_reference(language: str, override_b64: str = None) -> bytes:
    """
    Get voice reference audio bytes.
    Priority: override base64 → language WAV → hindi fallback
    """
    if override_b64:
        try:
            return base64.b64decode(override_b64)
        except Exception as e:
            print(f"Override audio decode failed: {e} — using default voice")

    lang_lower = language.lower() if language else "hindi"
    voice_map = {
        "hindi":    VOICE_HINDI,
        "hinglish": VOICE_HINDI,
        "hi":       VOICE_HINDI,
        "english":  VOICE_ENGLISH,
        "en":       VOICE_ENGLISH,
    }
    voice_path = voice_map.get(lang_lower, VOICE_HINDI)

    if not os.path.exists(voice_path):
        print(f"Voice file missing: {voice_path} — using Hindi fallback")
        voice_path = VOICE_HINDI

    with open(voice_path, "rb") as f:
        return f.read()


def decode_reference_audio(audio_base64: str) -> tuple:
    """Decode base64 audio and get duration info."""
    audio_bytes = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    try:
        waveform, sample_rate = torchaudio.load(temp_path)
        duration = waveform.shape[1] / sample_rate
        return audio_bytes, duration, sample_rate
    finally:
        os.unlink(temp_path)


def handler(job):
    """RunPod serverless handler for Fish Speech TTS."""
    global tts_engine

    try:
        job_input = job.get("input", {})

        # Text
        text = job_input.get("text") or job_input.get("prompt")
        if not text:
            return {"error": "Missing required parameter: text"}
        text = text.strip()
        if len(text) < MIN_TEXT_LENGTH:
            return {"error": f"Text too short. Minimum: {MIN_TEXT_LENGTH}"}
        if len(text) > MAX_TEXT_LENGTH:
            return {"error": f"Text too long. Maximum: {MAX_TEXT_LENGTH}"}

        print(f"Generating TTS for {len(text)} characters...")

        # Settings
        language          = job_input.get("language", "hindi")
        emotion_marker    = job_input.get("emotion_marker", NARRATION_STYLE)
        temperature       = max(0.1, min(1.0, float(job_input.get("temperature", 0.9))))
        top_p             = max(0.1, min(1.0, float(job_input.get("top_p", 0.85))))
        repetition_penalty = max(0.9, min(2.0, float(job_input.get("repetition_penalty", 1.1))))

        print(f"Language: {language}, emotion: '{emotion_marker}', temp: {temperature}")

        # Voice reference — use Pranshu's cloned voice from network volume
        reference_audio_b64 = job_input.get("reference_audio_base64")
        voice_bytes = get_voice_reference(language, reference_audio_b64)
        references = [ServeReferenceAudio(audio=voice_bytes, text="")]
        print(f"Voice reference loaded: {len(voice_bytes)} bytes")

        # Styled text
        styled_text = f"{emotion_marker} {text}" if emotion_marker.strip() else text
        print(f"Styled text: {styled_text[:100]}...")

        # TTS request
        request = ServeTTSRequest(
            text=styled_text,
            references=references,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=2048,
            normalize=True,
            format="wav",
            chunk_length=200,
            seed=None,
        )

        # Generate audio
        audio_segments = []
        result_sample_rate = None

        for result in tts_engine.inference(request):
            if hasattr(result, 'audio') and result.audio is not None:
                audio_data = result.audio
                if isinstance(audio_data, tuple) and len(audio_data) == 2:
                    sr, audio_np = audio_data
                    if result_sample_rate is None:
                        result_sample_rate = sr
                    if isinstance(audio_np, np.ndarray):
                        audio_segments.append(audio_np)
                    elif isinstance(audio_np, torch.Tensor):
                        audio_segments.append(audio_np.cpu().numpy())
                elif isinstance(audio_data, np.ndarray):
                    audio_segments.append(audio_data)
                elif isinstance(audio_data, torch.Tensor):
                    audio_segments.append(audio_data.cpu().numpy())
                else:
                    print(f"Warning: Unexpected audio type: {type(audio_data)}")

        if not audio_segments:
            return {"error": "No audio generated. Please check the input text."}

        # Concatenate and convert
        audio_np = np.concatenate(audio_segments, axis=0)
        print(f"Concatenated audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")

        audio = torch.from_numpy(audio_np).float()
        if audio.dim() > 1:
            audio = audio.squeeze()

        model_sample_rate = result_sample_rate or tts_engine.decoder_model.sample_rate
        print(f"Model sample rate: {model_sample_rate}, target: {OUTPUT_SAMPLE_RATE}")

        if model_sample_rate != OUTPUT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(
                audio.unsqueeze(0), model_sample_rate, OUTPUT_SAMPLE_RATE
            ).squeeze(0)

        audio = (audio * AMPLITUDE).clamp(-32768, 32767).to(torch.int16)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), OUTPUT_SAMPLE_RATE, format="wav")
        buffer.seek(0)
        audio_bytes = buffer.read()

        audio_base64_out = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"Generated {len(audio_bytes)} bytes ({len(audio_base64_out)} base64 chars)")

        return {
            "audio_base64": audio_base64_out,
            "sample_rate": OUTPUT_SAMPLE_RATE,
        }

    except Exception as e:
        error_msg = f"TTS generation failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}


# Cold start
print("Fish Speech TTS Handler starting...")
load_models()
runpod.serverless.start({"handler": handler})