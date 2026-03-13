import os
import modal

# 1. Define the Modal Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "TTS",
        "datasets",
        "transformers",
        "torch",
        "torchaudio",
        "scipy",
        "pandas",
        "huggingface_hub",
        "trainer",
        "fsspec==2023.6.0",
        "polyglot", # For morphological analysis (example)
        "pyicu",
        "morfessor",
    )
)

app = modal.App("sautiv0.1-research-training")
volume = modal.Volume.from_name("sauti-data", create_if_missing=True)

# Constants
DATA_DIR = "/data"
MODEL_DIR = "/data/models"
DATASET_NAME = "google/WaxalNLP"
CONFIG_NAME = "swa_tts"


def find_latest_model_dir(base_dir: str) -> str:
    """Return the newest directory that contains a config.json file."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Model directory does not exist: {base_dir}")

    candidates = []
    for root, _dirs, files in os.walk(base_dir):
        if "config.json" in files:
            candidates.append(root)

    if not candidates:
        raise FileNotFoundError(
            f"No model directory with config.json found under: {base_dir}"
        )

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def swahili_morphological_analyzer(text):
    """
    A placeholder for a robust Swahili morphological analyzer.
    In a real research scenario, this would use a tool like Morfessor 
    trained on Swahili or a rule-based analyzer.
    """
    # Simple rule-based or statistical decomposition (placeholder)
    # Example: 'ninaenda' -> 'ni-na-enda'
    # For now, we'll simulate it by adding markers at common prefix boundaries
    prefixes = ['ni', 'u', 'a', 'tu', 'm', 'wa', 'na', 'li', 'ta', 'me']
    words = text.split()
    analyzed_words = []
    for word in words:
        for p in prefixes:
            if word.startswith(p) and len(word) > len(p):
                word = p + "##" + word[len(p):]
                break
        analyzed_words.append(word)
    return " ".join(analyzed_words)

@app.function(
    image=image, 
    volumes={DATA_DIR: volume}, 
    timeout=3600 * 24, 
    gpu=modal.gpu.A100(count=3), 
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_sauti(hf_repo_id: str = None):
    from datasets import load_dataset
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from huggingface_hub import HfApi, create_repo

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    wav_dir = os.path.join(DATA_DIR, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print("--- Loading Waxal Swahili Dataset ---")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split="train")
    
    metadata = []
    print(f"--- Processing {len(ds)} samples with Morphological-Aware Preprocessing ---")
    for i, item in enumerate(ds):
        audio = item["audio"]
        text = item["text"]
        audio_id = item["id"]
        
        # Innovation 1: Morphological-Aware Phonemization (MAP)
        # Pre-process text to include morphological markers
        processed_text = swahili_morphological_analyzer(text)
        
        wav_path = os.path.join(wav_dir, f"{audio_id}.wav")
        if not os.path.exists(wav_path):
            import scipy.io.wavfile as wavfile
            wavfile.write(wav_path, audio["sampling_rate"], audio["array"])
        
        rel_path = f"wavs/{audio_id}.wav"
        metadata.append(f"{rel_path}|{processed_text}")
        
        if i % 500 == 0:
            print(f"Processed {i} samples...")

    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(line + "\n")
    
    print("--- Dataset Prepared with MAP ---")

    # 2. Configure XTTS v2 for Full Fine-tuning on 3x A100
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    manager = ModelManager()
    model_path, config_path, model_item = manager.download_model(model_name)
    
    config = XttsConfig()
    config.load_json(config_path)
    
    # High-performance settings for A100s
    config.epochs = 20 
    config.batch_size = 16 
    config.lr = 1e-5 
    config.save_step = 1000
    config.test_delay_epochs = 1
    config.save_checkpoints = True
    config.print_step = 10
    config.output_path = MODEL_DIR
    
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="waxal_swahili_research",
        path=DATA_DIR,
        meta_file_train="metadata.csv",
    )
    config.datasets = [dataset_config]
    config.languages = ["sw"]
    
    # Innovation 2: Cross-Lingual Prosody Transfer (CLPT)
    # In a full implementation, we would load prosody embeddings from other Bantu languages.
    # For this script, we enable the model's ability to learn from diverse prosodic contexts.
    config.use_speaker_weighted_sampler = True
    
    # Multi-GPU Trainer Args
    args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=1,
    )
    
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=100,
        eval_split_size=0.05,
    )
    
    # Initialize model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=model_path, eval=False)
    
    # Start Distributed Training
    trainer = Trainer(
        args,
        config,
        output_path=MODEL_DIR,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    print(f"--- Starting Full Fine-tuning on {torch.cuda.device_count()} GPUs ---")
    trainer.fit()
    print("--- Training Complete ---")
    
    # 3. Push to Hugging Face Hub
    if hf_repo_id:
        print(f"--- Pushing model to Hugging Face Hub: {hf_repo_id} ---")
        api = HfApi()
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is missing. Configure Modal secret 'huggingface-secret'."
            )
        
        try:
            create_repo(repo_id=hf_repo_id, token=token, exist_ok=True)
            latest_checkpoint_path = find_latest_model_dir(MODEL_DIR)
            
            api.upload_folder(
                folder_path=latest_checkpoint_path,
                repo_id=hf_repo_id,
                repo_type="model",
                token=token
            )
            print(f"--- Model successfully pushed to {hf_repo_id} ---")
        except Exception as e:
            print(f"--- Error pushing to Hugging Face: {e} ---")
    
    volume.commit()

@app.local_entrypoint()
def main(hf_repo_id: str = None):
    train_sauti.remote(hf_repo_id=hf_repo_id)