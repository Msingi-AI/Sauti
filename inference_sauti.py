import os
import modal

# 1. Define the Modal Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "TTS",
        "torch",
        "torchaudio",
        "scipy",
        "huggingface_hub",
    )
)

app = modal.App("sautiv0.1-inference")
volume = modal.Volume.from_name("sauti-data")

# Constants
DATA_DIR = "/data"
MODEL_DIR = "/data/models"


def resolve_model_dir(base_path: str) -> str:
    """Resolve a usable XTTS model directory containing config.json."""
    config_here = os.path.join(base_path, "config.json")
    if os.path.isfile(config_here):
        return base_path

    candidates = []
    for root, _dirs, files in os.walk(base_path):
        if "config.json" in files:
            candidates.append(root)

    if not candidates:
        raise FileNotFoundError(
            f"No model directory with config.json found under: {base_path}"
        )

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def pick_reference_wav(wav_dir: str) -> str:
    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(
            f"Reference audio directory not found: {wav_dir}. Run training first."
        )

    ref_wavs = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    if not ref_wavs:
        raise FileNotFoundError(
            f"No reference audio found in: {wav_dir}. Run training first."
        )

    ref_wavs.sort()
    return os.path.join(wav_dir, ref_wavs[0])

@app.function(
    image=image, 
    volumes={DATA_DIR: volume}, 
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")] # User must create this secret in Modal
)
def generate_swahili_speech(text: str, hf_repo_id: str = None, output_filename: str = "output.wav"):
    from TTS.api import TTS
    from huggingface_hub import snapshot_download

    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string")
    
    # Load the fine-tuned model
    if hf_repo_id:
        print(f"--- Loading Fine-tuned Model from Hugging Face: {hf_repo_id} ---")
        model_base_path = snapshot_download(
            repo_id=hf_repo_id,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        # Find the latest checkpoint in MODEL_DIR
        model_base_path = MODEL_DIR

    model_path = resolve_model_dir(model_base_path)
    print(f"--- Loading Fine-tuned Model from: {model_path} ---")
    
    # XTTS v2 requires a reference audio for voice cloning/style
    # We can use one of the Waxal samples as a reference
    wav_dir = os.path.join(DATA_DIR, "wavs")
    ref_wav_path = pick_reference_wav(wav_dir)
    
    tts = TTS(model_path=model_path, config_path=os.path.join(model_path, "config.json"), gpu=True)
    
    print(f"--- Generating Speech for: {text} ---")
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_wav_path,
        language="sw",
        file_path=output_filename
    )
    
    # Read the generated file to return it
    with open(output_filename, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(hf_repo_id: str = None):
    sample_text = "Habari gani? Karibu kwenye sautiv0.1, mfumo wa kisasa wa kutoa sauti kwa lugha ya Kiswahili."
    print(f"Generating speech for: {sample_text}")
    
    # Example: modal run inference_sauti.py --hf-repo-id "your-username/sautiv0.1"
    audio_bytes = generate_swahili_speech.remote(text=sample_text, hf_repo_id=hf_repo_id)
    
    output_path = "sautiv0.1_sample.wav"
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    
    print(f"--- Sample saved to {output_path} ---")