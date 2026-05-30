"""
Fish Speech TTS RunPod Serverless Handler
==========================================
Calls Fish Speech HTTP API (port 8080) — no direct Python imports.
Eliminates all module compatibility issues between image versions.

Input:
{
    "text": "Text to synthesize",
    "language": "hindi" | "english",
    "emotion_marker": "(sincere)",
    "temperature": 0.9,
    "top_p": 0.85,
    "repetition_penalty": 1.1
}

Output:
{
    "audio_base64": "base64 WAV audio",
    "sample_rate": 44100
}
"""

import runpod
import base64
import os
import time
import subprocess
import traceback
import threading
import shutil

import requests

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH    = os.environ.get("LLAMA_CHECKPOINT_PATH",   "/app/checkpoints/openaudio-s1-mini")
DECODER_CHECKPOINT = os.environ.get("DECODER_CHECKPOINT_PATH", f"{CHECKPOINT_PATH}/codec.pth")
DECODER_CONFIG     = os.environ.get("DECODER_CONFIG_NAME",     "modded_dac_vq")
VOICE_HINDI        = os.environ.get("VOICE_HINDI",   "/runpod-volume/fish-audio/voices/hindi/reference_hindi.wav")
VOICE_ENGLISH      = os.environ.get("VOICE_ENGLISH", "/runpod-volume/fish-audio/voices/english/reference_english.wav")
COMPILE            = os.environ.get("COMPILE", "0") == "1"
API_PORT           = int(os.environ.get("API_SERVER_PORT", "8080"))
API_URL            = f"http://127.0.0.1:{API_PORT}"
NARRATION_STYLE    = "(sincere) (soft tone)"

# References directory — Fish Speech serves voices from here
REFERENCES_DIR     = "/app/references"


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def verify_and_setup():
    """Verify volume paths and copy voice references."""
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

    # Copy voice references to /app/references/ so API server can find them
    os.makedirs(REFERENCES_DIR, exist_ok=True)
    shutil.copy2(VOICE_HINDI,   f"{REFERENCES_DIR}/hindi.wav")
    shutil.copy2(VOICE_ENGLISH, f"{REFERENCES_DIR}/english.wav")

    print(f"✅ Volume verified and references copied")
    print(f"   Checkpoint : {CHECKPOINT_PATH}")
    print(f"   Decoder    : {DECODER_CHECKPOINT}")
    print(f"   Hindi ref  : {REFERENCES_DIR}/hindi.wav")
    print(f"   English ref: {REFERENCES_DIR}/english.wav")
    print(f"   Device     : cuda if available")


def start_fish_server():
    """Start Fish Speech API server as background process."""
    cmd = [
        "/app/.venv/bin/python", "-m", "tools.api_server",
        "--listen", f"0.0.0.0:{API_PORT}",
        "--llama-checkpoint-path", CHECKPOINT_PATH,
        "--decoder-checkpoint-path", DECODER_CHECKPOINT,
        "--decoder-config-name", DECODER_CONFIG,
    ]
    if COMPILE:
        cmd.append("--compile")

    print(f"Starting Fish Speech API: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    def log_output():
        for line in proc.stdout:
            print(f"[FishAPI] {line.rstrip()}")
    threading.Thread(target=log_output, daemon=True).start()
    return proc


def wait_for_server(timeout: int = 180) -> bool:
    """Poll until Fish Speech API is ready."""
    print(f"Waiting for Fish Speech API (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_URL}/docs", timeout=3)
            if r.status_code in (200, 404):  # 404 is fine — server is up
                print(f"✅ API ready after {int(time.time()-start)}s")
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# Cold start
verify_and_setup()
fish_proc = start_fish_server()
if not wait_for_server():
    raise RuntimeError("Fish Speech API server failed to start")


# ══════════════════════════════════════════════════════════════════════════════
#  HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def handler(job):
    """RunPod handler — calls Fish Speech HTTP API."""
    try:
        inp = job.get("input", {})

        text = (inp.get("text") or inp.get("prompt") or "").strip()
        if not text:
            return {"error": "Missing required field: text"}
        if len(text) > 2000:
            return {"error": "Text too long. Max 2000 chars."}

        language           = inp.get("language", "hindi").lower()
        emotion_marker     = inp.get("emotion_marker", NARRATION_STYLE)
        temperature        = max(0.1, min(1.0, float(inp.get("temperature", 0.9))))
        top_p              = max(0.1, min(1.0, float(inp.get("top_p", 0.85))))
        repetition_penalty = max(0.9, min(2.0, float(inp.get("repetition_penalty", 1.1))))

        print(f"TTS: {len(text)} chars, language={language}")

        # Select voice reference ID
        if language in ("hindi", "hinglish", "hi"):
            reference_id = "hindi"
        else:
            reference_id = "english"

        # Styled text
        styled_text = f"{emotion_marker} {text}" if emotion_marker.strip() else text

        # Call Fish Speech API
        payload = {
            "text":               styled_text,
            "reference_id":       reference_id,
            "temperature":        temperature,
            "top_p":              top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens":     2048,
            "normalize":          True,
            "format":             "wav",
            "chunk_length":       200,
        }

        print(f"Calling {API_URL}/v1/tts with reference_id={reference_id}...")
        resp = requests.post(
            f"{API_URL}/v1/tts",
            json=payload,
            timeout=120,
        )

        if resp.status_code != 200:
            return {"error": f"Fish API error {resp.status_code}: {resp.text[:300]}"}

        audio_bytes   = resp.content
        audio_b64_out = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"✅ Generated {len(audio_bytes)/1024:.0f}KB audio")

        return {
            "audio_base64": audio_b64_out,
            "sample_rate":  44100,
        }

    except Exception as e:
        print(f"Handler error: {e}")
        traceback.print_exc()
        return {"error": str(e)}


print("Fish Speech Handler ready — waiting for jobs!")
runpod.serverless.start({"handler": handler})
