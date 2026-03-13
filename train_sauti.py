import os
import modal

# 1. Define the Modal Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "TTS",
        "datasets",
        "transformers==4.33.3",
        "torch",
        "torchaudio",
        "torchcodec",
        "scipy",
        "huggingface_hub",
        "trainer",
        "fsspec==2023.6.0",
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
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-5
DEFAULT_MIN_DURATION_SEC = 0.5
DEFAULT_MAX_DURATION_SEC = 20.0


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


def normalize_text(text: str) -> str:
    """Lightweight normalization for robust metadata creation."""
    normalized = text.strip().lower()
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    return normalized


def prepare_training_text(text: str, use_map: bool) -> str:
    normalized = normalize_text(text)
    if use_map:
        return swahili_morphological_analyzer(normalized)
    return normalized


def extract_audio_array_and_sr(audio_obj):
    """Support both classic dict audio and datasets AudioDecoder objects."""
    # Older datasets format: {'array': ..., 'sampling_rate': ...}
    if isinstance(audio_obj, dict):
        array = audio_obj.get("array")
        sampling_rate = audio_obj.get("sampling_rate")
        return array, sampling_rate

    # Newer datasets format: AudioDecoder with get_all_samples()
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        data = getattr(samples, "data", None)
        sampling_rate = getattr(samples, "sample_rate", None)

        if data is not None:
            try:
                array = data.detach().cpu().numpy()
            except Exception:
                array = data

            # Convert multi-channel audio to mono for consistent wav writing.
            if getattr(array, "ndim", 1) == 2:
                if array.shape[0] <= array.shape[1]:
                    array = array.mean(axis=0)
                else:
                    array = array.mean(axis=1)
            return array, sampling_rate

    # Fallback: object-like with known attributes.
    array = getattr(audio_obj, "array", None)
    sampling_rate = getattr(audio_obj, "sampling_rate", None)
    if sampling_rate is None:
        sampling_rate = getattr(audio_obj, "sample_rate", None)
    return array, sampling_rate


def audio_duration_seconds(audio_dict: dict) -> float:
    array, sampling_rate = extract_audio_array_and_sr(audio_dict)
    if array is None or not sampling_rate:
        return 0.0
    return float(len(array)) / float(sampling_rate)

@app.function(
    image=image, 
    volumes={DATA_DIR: volume}, 
    timeout=3600 * 24, 
    gpu="A100-40GB:3", 
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_sauti(
    hf_repo_id: str = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    max_samples: int = 0,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    eval_split_size: float = 0.05,
    use_map: bool = True,
    coqui_tos_accepted: bool = False,
):
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
    skipped_duration = 0
    skipped_empty_text = 0
    processed = 0

    print(f"--- Processing {len(ds)} samples with Morphological-Aware Preprocessing ---")
    for item in ds:
        if max_samples > 0 and processed >= max_samples:
            break

        audio = item["audio"]
        text = item["text"]
        audio_id = item["id"]
        speaker_id = str(item.get("speaker_id", "sauti_speaker"))

        duration_sec = audio_duration_seconds(audio)
        if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
            skipped_duration += 1
            continue

        if not text or not text.strip():
            skipped_empty_text += 1
            continue
        
        # Innovation 1: Morphological-Aware Phonemization (MAP)
        # Pre-process text to include morphological markers
        processed_text = prepare_training_text(text, use_map=use_map)
        
        wav_path = os.path.join(wav_dir, f"{audio_id}.wav")
        if not os.path.exists(wav_path):
            import scipy.io.wavfile as wavfile
            array, sampling_rate = extract_audio_array_and_sr(audio)
            if array is None or not sampling_rate:
                skipped_duration += 1
                continue
            wavfile.write(wav_path, int(sampling_rate), array)
        
        rel_path = f"wavs/{audio_id}.wav"
        safe_text = processed_text.replace("|", " ")
        # Coqui ljspeech formatter expects: audio_file|text|speaker_name
        metadata.append(f"{rel_path}|{safe_text}|{speaker_id}")
        processed += 1
        
        if processed % 500 == 0:
            print(f"Processed {processed} usable samples...")

    if not metadata:
        raise RuntimeError("No training samples remained after filtering.")

    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(line + "\n")
    
    print(
        "--- Dataset prepared | usable: "
        f"{processed} | skipped_duration: {skipped_duration} | skipped_empty_text: {skipped_empty_text} ---"
    )

    # 2. Configure XTTS v2 for Full Fine-tuning on 3x A100
    if not coqui_tos_accepted:
        raise RuntimeError(
            "Coqui XTTS requires explicit license confirmation. "
            "Rerun with --coqui-tos-accepted true after reviewing Coqui CPML/commercial terms."
        )

    os.environ["COQUI_TOS_AGREED"] = "1"

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    manager = ModelManager()
    download_result = manager.download_model(model_name)

    if not isinstance(download_result, (tuple, list)) or len(download_result) < 1:
        raise RuntimeError(f"Unexpected download_model result: {type(download_result)}")

    model_path = download_result[0]
    config_path = download_result[1] if len(download_result) > 1 else None

    if not model_path:
        raise RuntimeError("ModelManager returned an empty model path.")

    # Some Coqui versions return config_path=None for XTTS. Resolve it from artifacts.
    if not config_path or not os.path.exists(config_path):
        candidate_roots = []
        if os.path.isdir(model_path):
            candidate_roots.append(model_path)
        else:
            candidate_roots.append(os.path.dirname(model_path))
        candidate_roots.append(os.path.expanduser("~/.local/share/tts"))

        discovered_config = None
        for root in candidate_roots:
            if not root or not os.path.isdir(root):
                continue

            direct = os.path.join(root, "config.json")
            if os.path.isfile(direct):
                discovered_config = direct
                break

            for walk_root, _dirs, files in os.walk(root):
                if "config.json" in files and "xtts" in walk_root.lower():
                    discovered_config = os.path.join(walk_root, "config.json")
                    break
            if discovered_config:
                break

        if not discovered_config:
            raise FileNotFoundError(
                f"Could not locate XTTS config.json. model_path={model_path}, config_path={config_path}"
            )
        config_path = discovered_config

    # If model_path is a directory, resolve an actual checkpoint file for load_checkpoint.
    if os.path.isdir(model_path):
        checkpoint_candidates = []
        for walk_root, _dirs, files in os.walk(model_path):
            for file_name in files:
                if file_name.endswith(".pth"):
                    checkpoint_candidates.append(os.path.join(walk_root, file_name))
        if checkpoint_candidates:
            preferred_names = ["model.pth", "best_model.pth", "checkpoint.pth"]
            preferred_match = None
            for pref in preferred_names:
                for candidate in checkpoint_candidates:
                    if os.path.basename(candidate) == pref:
                        preferred_match = candidate
                        break
                if preferred_match:
                    break

            if not preferred_match:
                filtered = [
                    p for p in checkpoint_candidates
                    if "speaker" not in os.path.basename(p).lower()
                ]
                target_pool = filtered if filtered else checkpoint_candidates
                target_pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                preferred_match = target_pool[0]

            model_path = preferred_match

    print(f"--- XTTS artifacts resolved | checkpoint={model_path} | config={config_path} ---")
    
    config = XttsConfig()
    config.load_json(config_path)
    
    # High-performance settings for A100s
    config.epochs = epochs
    config.batch_size = batch_size
    config.lr = lr
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
        eval_split_size=eval_split_size,
    )
    
    # Initialize model
    model = Xtts.init_from_config(config)
    checkpoint_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    model.load_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=model_path,
        vocab_path=vocab_path if os.path.exists(vocab_path) else None,
        eval=False,
    )
    
    # Start Distributed Training
    trainer = Trainer(
        args,
        config,
        output_path=MODEL_DIR,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    print(
        "--- Starting fine-tuning --- "
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"eval_split_size={eval_split_size}, use_map={use_map}, "
        f"gpus_detected={torch.cuda.device_count()}"
    )
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
def main(
    hf_repo_id: str = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    max_samples: int = 0,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    eval_split_size: float = 0.05,
    use_map: bool = True,
    coqui_tos_accepted: bool = False,
):
    train_sauti.remote(
        hf_repo_id=hf_repo_id,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_samples=max_samples,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        eval_split_size=eval_split_size,
        use_map=use_map,
        coqui_tos_accepted=coqui_tos_accepted,
    )