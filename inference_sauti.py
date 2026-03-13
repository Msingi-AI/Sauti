import os
import modal
from pathlib import Path

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

@app.function(
    image=image, 
    volumes={DATA_DIR: volume}, 
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")] # User must create this secret in Modal
)
def generate_swahili_speech(text: str, hf_repo_id: str = None, output_filename: str = "output.wav"):
    from TTS.api import TTS
    import torch
    from huggingface_hub import snapshot_download
    
    # Load the fine-tuned model
    if hf_repo_id:
        print(f"--- Loading Fine-tuned Model from Hugging Face: {hf_repo_id} ---")
        model_path = snapshot_download(repo_id=hf_repo_id, token=os.environ.get("HF_TOKEN"))
    else:
        # Find the latest checkpoint in MODEL_DIR
        checkpoint_dirs = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
        if not checkpoint_dirs:
            raise ValueError("No fine-tuned model found in MODEL_DIR")
        
        latest_dir = sorted(checkpoint_dirs)[-1]
        model_path = os.path.join(MODEL_DIR, latest_dir)
        print(f"--- Loading Fine-tuned Model from local volume: {model_path} ---")
    
    # XTTS v2 requires a reference audio for voice cloning/style
    # We can use one of the Waxal samples as a reference
    wav_dir = os.path.join(DATA_DIR, "wavs")
    ref_wavs = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    if not ref_wavs:
        raise ValueError("No reference audio found in wav_dir")
    
    ref_wav_path = os.path.join(wav_dir, ref_wavs[0])
    
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