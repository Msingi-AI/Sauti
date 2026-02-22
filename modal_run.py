import os
import modal

# 1. THE APP
app = modal.App("sauti-distill")

# 2. THE VOLUME
sauti_volume = modal.Volume.from_name("sauti-volume", create_if_missing=True)

# 3. THE IMAGE (Updated for proper imports)
image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libsndfile1", "ffmpeg", "git", "build-essential")
        # Ensure modern build tooling to avoid wheel/build surprises
        .pip_install("pip", "setuptools", "wheel")
        # Layer 1: install tokenizer backends and foundations first (pin protobuf)
        .pip_install(
            "numpy", "scipy", "pyyaml", "wandb",
            "sentencepiece", "tiktoken", "protobuf>=4.25,<6"
        )
        # Layer 2: install transformers and heavier ML stack
        .pip_install(
            "torch", "torchaudio", "transformers>=4.45.0", "datasets", "accelerate", "huggingface-hub",
            "librosa", "soundfile", "tokenizers", "torchcodec"
        )
        # Run quick smoke checks at build-time to fail-fast on missing backends
        .run_commands(
            "python -c \"import importlib.util as iu; print('sentencepiece', bool(iu.find_spec('sentencepiece'))); print('tiktoken', bool(iu.find_spec('tiktoken')));\"",
            "python -c \"import transformers; print('transformers', transformers.__version__)\"",
        )
        .add_local_python_source("src")
        .add_local_file("run_distill.py", remote_path="/root/run_distill.py")
        .add_local_dir("configs", remote_path="/root/configs")
)

VOLUME_PATH = "/root/data"

@app.function(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("hf-token"), 
        modal.Secret.from_name("wandb")
    ],
    volumes={VOLUME_PATH: sauti_volume},
    timeout=3600
)
def precompute_max_items(max_items: int = 2000):
        # Ensure mounted src is importable
        import sys
        for p in ["/root/src", "/root/src/sauti", "/root/project", "/root"]:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        # Configure HuggingFace tokens (Modal secret may set HF_TOKEN or hf-token)
        if "HF_TOKEN" in os.environ:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        if "hf-token" in os.environ:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["hf-token"]

        # Persist HF datasets cache to the mounted volume so downloads survive
        # across runs and decoding can read local files reliably.
        os.environ["HF_DATASETS_CACHE"] = f"{VOLUME_PATH}/hf_cache"
        os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

        from sauti.precompute import precompute_teacher_activations_whisper as precompute_teacher_activations

        out_dir = f"{VOLUME_PATH}/teacher_activations"
        os.makedirs(out_dir, exist_ok=True)

        print(f"🚀 Initializing Whisper Teacher Inference. Target: {max_items} items.")
        print(f"📂 HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")
        # Run precompute (non-streaming, dataset will be downloaded into the HF cache)
        precompute_teacher_activations("google/WaxalNLP", "swa_tts", out_dir=out_dir, max_items=max_items, count_stream=True)

        try:
            sauti_volume.commit()
            print("✅ Volume committed. Data is safe.")
        except Exception:
            print("⚠️ Volume commit not supported or failed.")