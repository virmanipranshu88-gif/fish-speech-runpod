"""
Fish Speech TTS RunPod Serverless Handler

Provides voice cloning TTS using Fish Speech OpenAudio S1-mini model.
Compatible with the existing HistoryGen AI audio generation pipeline.

Input format (same as ChatterboxTTS):
{
    "text": "The text to synthesize",
    "reference_audio_base64": "base64 encoded audio for voice cloning",
    "emotion_marker": "(sincere) (soft tone)",  # Optional emotion/tone marker
    "temperature": 0.9,  # Optional: 0.1-1.0, higher = more expressive
    "top_p": 0.85,  # Optional: 0.1-1.0, higher = more variation
    "repetition_penalty": 1.1  # Optional: 0.9-2.0, higher = less repetition
}

Output format:
{
    "audio_base64": "base64 encoded WAV audio",
    "sample_rate": 24000
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

# Verify model is present (should be baked into Docker image)
def verify_model_present():
    """Verify model weights are present (baked into Docker image at build time)."""
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/openaudio-s1-mini")
    if not os.path.exists(checkpoint_path) or not os.listdir(checkpoint_path):
        raise RuntimeError(
            f"Model not found at {checkpoint_path}. "
            "Model should be baked into Docker image during build. "
            "Rebuild with: docker build --build-arg HF_TOKEN=your_token ."
        )
    print(f"Model present at {checkpoint_path}")

verify_model_present()

# Fish Speech imports (after model download)
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# Configuration
CHECKPOINT_PATH    = os.environ.get("LLAMA_CHECKPOINT_PATH", "/app/checkpoints/openaudio-s1-mini")
DECODER_CHECKPOINT = os.environ.get("DECODER_CHECKPOINT_PATH", f"{CHECKPOINT_PATH}/codec.pth")
DECODER_CONFIG     = os.environ.get("DECODER_CONFIG_NAME", "modded_dac_vq")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()
COMPILE_MODEL = os.environ.get("COMPILE_MODEL", "true").lower() == "true"

# Constants
MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 2000
MIN_REFERENCE_DURATION = 3.0
OUTPUT_SAMPLE_RATE = 24000
AMPLITUDE = 32768  # Scale for 16-bit PCM

# Emotion/tone markers for expressive speech
# These are prepended to text to make Fish Speech output more natural and engaging
# See: https://speech.fish.audio/ for full list
DEFAULT_EMOTION_MARKER = "(engaging)"  # General marker for documentary narration
NARRATION_STYLE = "(sincere) (soft tone)"  # Warm, documentary-style delivery

# Global engine instance
tts_engine = None


def load_models():
    """Load Fish Speech models on cold start."""
    global tts_engine

    print(f"Loading Fish Speech models from {CHECKPOINT_PATH}...")
    print(f"Device: {DEVICE}, Half precision: {HALF_PRECISION}, Compile: {COMPILE_MODEL}")

    # Determine precision
    precision = torch.float16 if HALF_PRECISION else torch.bfloat16

    # Load LLAMA (text-to-semantic) model queue
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        precision=precision,
        compile=COMPILE_MODEL,
    )
    print("LLAMA model loaded")

    # Load decoder (semantic-to-audio) model
    decoder_model = load_decoder_model(
        config_name=DECODER_CONFIG,
        checkpoint_path=DECODER_CHECKPOINT,
        device=DEVICE,
    )
    print(f"Decoder model loaded, sample rate: {decoder_model.sample_rate}")

    # Create TTS inference engine
    tts_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=COMPILE_MODEL,
    )
    print("TTS inference engine created")

    # Warm up the models
    print("Warming up models...")
    try:
        warmup_request = ServeTTSRequest(
            text="Hello, this is a warmup test.",
            references=[],
            temperature=0.7,
            repetition_penalty=1.2,
            format="wav",
        )
        # Run warmup inference
        audio_chunks = []
        for result in tts_engine.inference(warmup_request):
            if hasattr(result, 'audio') and result.audio is not None:
                audio_chunks.append(result.audio)
        print("Models warmed up successfully")
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

    print("Fish Speech models ready!")


def decode_reference_audio(audio_base64: str) -> tuple:
    """Decode base64 audio and get duration info."""
    audio_bytes = base64.b64decode(audio_base64)

    # Write to temp file for torchaudio to read
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
    """
    RunPod serverless handler for Fish Speech TTS.

    Input:
        text (str): Text to synthesize
        reference_audio_base64 (str, optional): Base64 encoded audio for voice cloning

    Output:
        audio_base64 (str): Base64 encoded WAV audio
        sample_rate (int): Audio sample rate (24000)
    """
    global tts_engine

    try:
        job_input = job.get("input", {})

        # Get text (support both "text" and "prompt" keys for compatibility)
        text = job_input.get("text") or job_input.get("prompt")
        if not text:
            return {"error": "Missing required parameter: text"}

        # Validate text length
        text = text.strip()
        if len(text) < MIN_TEXT_LENGTH:
            return {"error": f"Text too short. Minimum length: {MIN_TEXT_LENGTH}"}
        if len(text) > MAX_TEXT_LENGTH:
            return {"error": f"Text too long. Maximum length: {MAX_TEXT_LENGTH}"}

        print(f"Generating TTS for {len(text)} characters...")

        # Extract TTS settings from input (with defaults)
        emotion_marker = job_input.get("emotion_marker", NARRATION_STYLE)
        temperature = job_input.get("temperature", 0.9)
        top_p = job_input.get("top_p", 0.85)
        repetition_penalty = job_input.get("repetition_penalty", 1.1)

        # Validate and clamp settings
        temperature = max(0.1, min(1.0, float(temperature)))
        top_p = max(0.1, min(1.0, float(top_p)))
        repetition_penalty = max(0.9, min(2.0, float(repetition_penalty)))

        print(f"TTS settings: emotion='{emotion_marker}', temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}")

        # Process reference audio for voice cloning
        references = []
        reference_audio_base64 = job_input.get("reference_audio_base64")

        if reference_audio_base64:
            print("Processing reference audio for voice cloning...")
            try:
                audio_bytes, duration, sample_rate = decode_reference_audio(reference_audio_base64)
                print(f"Reference audio: {duration:.1f}s, {sample_rate}Hz")

                if duration < MIN_REFERENCE_DURATION:
                    print(f"Warning: Reference audio is short ({duration:.1f}s). Recommend 10-30s.")

                # Create reference audio object
                references.append(
                    ServeReferenceAudio(
                        audio=audio_bytes,
                        text="",  # Empty text for zero-shot cloning
                    )
                )
            except Exception as e:
                print(f"Error processing reference audio: {e}")
                traceback.print_exc()
                # Continue without voice cloning
                references = []

        # Add emotion markers for expressive narration
        # Fish Speech supports markers like (excited), (soft tone), (sincere) etc.
        # We prepend style markers to make documentary narration more engaging
        if emotion_marker and emotion_marker.strip():
            styled_text = f"{emotion_marker} {text}"
        else:
            styled_text = text
        print(f"Styled text: {styled_text[:100]}...")

        # Build TTS request using extracted settings
        request = ServeTTSRequest(
            text=styled_text,
            references=references,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=2048,
            normalize=True,
            format="wav",
            chunk_length=200,      # Default chunk size for better prosody
            seed=None,             # Random seed for natural variation
        )

        # Generate audio using inference engine
        # Fish Speech yields InferenceResult with audio = (sample_rate, numpy_array)
        audio_segments = []
        result_sample_rate = None

        for result in tts_engine.inference(request):
            if hasattr(result, 'audio') and result.audio is not None:
                audio_data = result.audio

                # Fish Speech returns (sample_rate, numpy_array) tuple
                if isinstance(audio_data, tuple) and len(audio_data) == 2:
                    sr, audio_np = audio_data
                    if result_sample_rate is None:
                        result_sample_rate = sr
                    # audio_np is a numpy array (float32)
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

        # Concatenate all numpy segments
        audio_np = np.concatenate(audio_segments, axis=0)
        print(f"Concatenated audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")

        # Convert to tensor for processing
        audio = torch.from_numpy(audio_np).float()

        # Ensure 1D
        if audio.dim() > 1:
            audio = audio.squeeze()

        # Get sample rate (from result or decoder model)
        model_sample_rate = result_sample_rate or tts_engine.decoder_model.sample_rate
        print(f"Model sample rate: {model_sample_rate}, target: {OUTPUT_SAMPLE_RATE}")

        # Resample to output sample rate if needed
        if model_sample_rate != OUTPUT_SAMPLE_RATE:
            audio = audio.unsqueeze(0)  # Add channel dim for resample
            audio = torchaudio.functional.resample(audio, model_sample_rate, OUTPUT_SAMPLE_RATE)
            audio = audio.squeeze(0)

        # Convert to int16
        audio = (audio * AMPLITUDE).clamp(-32768, 32767).to(torch.int16)

        # Ensure correct shape for torchaudio (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Save to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), OUTPUT_SAMPLE_RATE, format="wav")
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Encode to base64
        audio_base64_out = base64.b64encode(audio_bytes).decode("utf-8")

        print(f"Generated {len(audio_bytes)} bytes of audio ({len(audio_base64_out)} base64 chars)")

        return {
            "audio_base64": audio_base64_out,
            "sample_rate": OUTPUT_SAMPLE_RATE,
        }

    except Exception as e:
        error_msg = f"TTS generation failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}


# Load models on cold start
print("Fish Speech TTS Handler starting...")
load_models()

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
