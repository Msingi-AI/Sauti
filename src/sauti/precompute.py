import logging
from pathlib import Path

import torch
from tqdm import tqdm
from .data import get_waxal_swahili
from .distill_pipeline import append_manifest_record, deterministic_sample_id
from .security import is_remote_code_trusted

logger = logging.getLogger(__name__)


def precompute_teacher_activations(dataset_name, config_name, out_dir, max_items=2000, manifest_filename="manifest.jsonl"):
    """Run the teacher model on dataset samples and persist hidden states + manifest."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Teacher (Fish Speech 1.5)
    TEACHER_ID = "fishaudio/fish-speech-1.5"
    logger.info(f"Loading Teacher: {TEACHER_ID}")
    
    # Runtime checks: show whether tokenizer backends are importable in this
    # container. This helps distinguish image build vs runtime failures.
    try:
        import importlib.util as iu
        logger.info("sentencepiece available: %s", bool(iu.find_spec("sentencepiece")))
        logger.info("tiktoken available: %s", bool(iu.find_spec("tiktoken")))
    except Exception:
        logger.debug("Could not run importlib checks for tokenizer backends.")

    # USE_FAKE_ACTIVATIONS env var allows forcing synthetic activations for
    # smoke runs or debugging without heavy model downloads. Default is off.
    fake_env = os.environ.get("USE_FAKE_ACTIVATIONS", "0").lower() in ("1", "true", "yes")

    # Instantiate tokenizer with a fallback: some environments lack fast-tokenizer
    # backends at import time; try the default first, then fall back to the
    # slow Python tokenizer (`use_fast=False`) if needed. If USE_FAKE_ACTIVATIONS
    # is set, skip actual model loading entirely. Import `transformers` lazily
    # so local smoke tests can run without the heavy dependency installed.
    fake_mode = False
    tokenizer = None
    teacher = None
    if fake_env:
        logger.info("USE_FAKE_ACTIVATIONS enabled: skipping model download and using synthetic activations.")
        fake_mode = True
    else:
        try:
            try:
                from transformers import AutoModel, AutoTokenizer
            except Exception as imp_err:
                logger.warning("Transformers not available or failed to import (%s). Falling back to synthetic activations.", imp_err)
                fake_mode = True

            if not fake_mode:
                try:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
                    except ValueError as ve:
                        logger.warning("Fast tokenizer backend unavailable, falling back to slow tokenizer: %s", ve)
                        tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True, use_fast=False)

                    teacher = AutoModel.from_pretrained(TEACHER_ID, trust_remote_code=True).to(device)
                    teacher.eval()
                except Exception as load_err:
                    # Don't silently fallback here — surface a helpful error so the
                    # Modal image or local environment can be fixed (install tokenizers,
                    # ensure HF token, correct transformers version, etc.).
                    msg = (
                        "Failed to load teacher model or tokenizer: {}.\n"
                        "Remediation: install/enable tokenizer backends (sentencepiece, tiktoken),\n"
                        "ensure `transformers` is available, and provide a valid HuggingFace token.\n"
                        "On Modal, add your token as a secret named `hf-token` and rebuild the image so\n"
                        "that `sentencepiece`/`tiktoken` are installed at image build time.\n"
                        "Original error: {}"
                    ).format(str(load_err), repr(load_err))
                    logger.error(msg)
                    raise RuntimeError(msg)
        except Exception:
            # Catch any unexpected import-related errors and fall back
            fake_mode = True

    # 2. Load Data
    ds = get_waxal_swahili(split="train", streaming=True)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # 3. Processing Loop
    logger.info(f"Processing {max_items} items...")
    count = 0

    with torch.no_grad():
        for sample in tqdm(ds):
            if count >= max_items:
                break

            text = sample.get('text', "")

            # Prefer any existing id-like field from the dataset, fall back to index
            uid_base = sample.get('id') or sample.get('uid') or sample.get('key') or f"sample_{i}"
            uid = str(uid_base).replace(os.sep, "_").replace(" ", "_")
            save_path = os.path.join(out_dir, f"{uid}.pt")

            # Skip if already exists (resume capability)
            if os.path.exists(save_path):
                continue

            if fake_mode and not fake_env:
                # If fake_mode was set due to a load error and the user did not request
                # fake activations explicitly, stop here so failures are visible.
                raise RuntimeError("Teacher model/tokenizer not available and USE_FAKE_ACTIVATIONS is not set.")

            if not fake_mode and tokenizer is not None and teacher is not None:
                # Tokenize & move tensors to device correctly
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = teacher(**inputs, output_hidden_states=True)

                # Extract the specific layer we want to distill (e.g., last layer)
                # Fish Speech specific: check architecture for correct layer
                hidden_states = outputs.last_hidden_state.cpu()
            else:
                # If we reach here, either the user explicitly requested fake activations
                # via `USE_FAKE_ACTIVATIONS`, or the dataset provides no text. In the
                # explicit fake case, create a small random tensor for smoke runs.
                if not fake_env:
                    raise RuntimeError("Unexpected state: neither real teacher nor fake activations available.")
                seq_len = 32
                hidden_dim = 768
                hidden_states = torch.randn(1, seq_len, hidden_dim)

            # Robustly extract audio path metadata
            audio_meta = sample.get('audio')
            audio_path = ""
            try:
                if isinstance(audio_meta, dict):
                    audio_path = audio_meta.get('path', '')
                elif isinstance(audio_meta, str):
                    audio_path = audio_meta
            except Exception:
                audio_path = ""

            # Save (ensure tensors are on CPU for portability)
            torch.save({
                "hidden_states": hidden_states.cpu() if isinstance(hidden_states, torch.Tensor) else hidden_states,
                "text": text,
                "audio_path": audio_path,
            }, save_path)

            count += 1

    logger.info(f"Precomputation complete. Saved {count} files to {out_dir}")


def precompute_teacher_activations_whisper(dataset_name, config_name, out_dir, max_items=2000, count_stream=False):
    # Diagnostic precompute: focus on discovering the exact structure of sample['audio']
    # and attempt simple bytes/path decoding for the first items.
    try:
        import io as _io
        import hashlib
        import soundfile as sf
        import librosa as _librosa
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as _e:
        logger.error("Required debug deps missing: %s", _e)
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting diagnostic audio loader (no Whisper forward).")

    def _stable_uid(sample, index):
        try:
            p = sample.get('audio', {}).get('path', '')
            if p:
                return hashlib.md5(p.encode()).hexdigest()[:12]
        except Exception:
            pass
        return f"sample_{index}"

    def load_audio_debug(audio_data, index):
        target_sr = 16000
        if audio_data is None:
            logger.warning("DEBUG: sample %d audio field is None", index)
            return None

        if isinstance(audio_data, dict):
            keys = list(audio_data.keys())
        else:
            try:
                keys = [str(type(audio_data))]
            except Exception:
                keys = []

        if index == 0:
            logger.warning("🔎 DEBUG: Sample 0 audio keys=%s", keys)
            try:
                if isinstance(audio_data, dict) and 'path' in audio_data:
                    logger.warning("   path=%s", audio_data.get('path'))
                if isinstance(audio_data, dict) and 'bytes' in audio_data:
                    b = audio_data.get('bytes')
                    logger.warning("   bytes type=%s len=%s", type(b), (len(b) if b is not None else None))
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    arr = audio_data.get('array')
                    logger.warning("   array type=%s shape=%s", type(arr), getattr(arr, 'shape', None))
            except Exception:
                logger.exception("Failed to pretty-print audio_data for sample 0")

        # Try bytes first
        try:
            if isinstance(audio_data, dict) and audio_data.get('bytes'):
                try:
                    wav, sr = sf.read(_io.BytesIO(audio_data.get('bytes')))
                    if int(sr) != target_sr:
                        wav = _librosa.resample(wav, orig_sr=int(sr), target_sr=target_sr)
                    return wav
                except Exception as e:
                    logger.warning("DEBUG: bytes decode failed for sample %d: %s", index, e)

            # Try path on disk
            path = None
            if isinstance(audio_data, dict):
                path = audio_data.get('path') or audio_data.get('file') or audio_data.get('audio_filepath')
            elif isinstance(audio_data, str):
                path = audio_data

            if path:
                logger.warning("DEBUG: sample %d path=%s exists=%s", index, path, os.path.exists(path))
                if os.path.exists(path):
                    try:
                        wav, sr = _librosa.load(path, sr=target_sr)
                        return wav
                    except Exception as e:
                        logger.warning("DEBUG: librosa failed on path for sample %d: %s", index, e)

            # Try handling datasets' lazy AudioDecoder (torchcodec) objects
            try:
                t = audio_data
                t_repr = repr(type(t))
                if 'AudioDecoder' in t_repr or 'torchcodec' in t_repr:
                    logger.warning("DEBUG: sample %d audio appears to be AudioDecoder (%s). Attempting methods.", index, t_repr)
                    # Try a list of common method names that may return arrays/bytes
                    for meth in ('decode', 'to_array', 'array', 'read', 'get_array', '__call__'):
                        try:
                            if hasattr(t, meth):
                                fn = getattr(t, meth)
                                res = fn() if callable(fn) else fn
                                # If result is dict with 'array'
                                if isinstance(res, dict) and res.get('array') is not None:
                                    arr = res.get('array')
                                    sr = int(res.get('sampling_rate', target_sr))
                                    if sr != target_sr:
                                        arr = _librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
                                    return arr
                                # If result is a numpy-like array or list
                                try:
                                    import numpy as _np
                                    if _np and (isinstance(res, _np.ndarray) or hasattr(res, 'shape') or isinstance(res, (list, tuple))):
                                        arr = _np.array(res)
                                        if arr.size:
                                            return arr
                                except Exception:
                                    pass
                                # If result is raw bytes
                                if isinstance(res, (bytes, bytearray)):
                                    try:
                                        wav, sr = sf.read(_io.BytesIO(res))
                                        if int(sr) != target_sr:
                                            wav = _librosa.resample(wav, orig_sr=int(sr), target_sr=target_sr)
                                        return wav
                                    except Exception as e:
                                        logger.warning("DEBUG: AudioDecoder bytes->soundfile failed: %s", e)
                        except Exception as _meth_e:
                            logger.debug("AudioDecoder method %s on sample %d raised: %s", meth, index, _meth_e)
                    # If object supports __getitem__ like datasets' AudioDecoder, try indexing
                    try:
                        if hasattr(t, '__getitem__'):
                            try:
                                arr = t['array']
                                sr = t['sampling_rate'] if 'sampling_rate' in dir(t) or True else target_sr
                                try:
                                    import numpy as _np
                                    arr_np = _np.array(arr)
                                except Exception:
                                    arr_np = arr
                                if arr_np is not None and getattr(arr_np, 'size', None) != 0:
                                    if int(sr) != target_sr:
                                        arr_np = _librosa.resample(arr_np, orig_sr=int(sr), target_sr=target_sr)
                                    return arr_np
                            except Exception as _idx_e:
                                logger.debug("AudioDecoder __getitem__ access failed for sample %d: %s", index, _idx_e)
                    except Exception:
                        pass
            except Exception:
                logger.debug("AudioDecoder probing failed for sample %d", index)

        except Exception as e:
            logger.exception("Unexpected debug loader exception for sample %d: %s", index, e)

        return None

    # Non-streaming dataset so files are downloaded and available locally when possible
    ds = get_waxal_swahili(split="train", streaming=False)
    os.makedirs(out_dir, exist_ok=True)

    # Load Whisper processor + model (encoder outputs as teacher activations)
    WHISPER_ID = os.environ.get("WHISPER_TEACHER", "openai/whisper-large-v3")
    try:
        from transformers import WhisperProcessor, WhisperModel
        logger.info("Loading Whisper teacher: %s", WHISPER_ID)
        processor = WhisperProcessor.from_pretrained(WHISPER_ID)
        whisper = WhisperModel.from_pretrained(WHISPER_ID).to(device)
        whisper.eval()
    except Exception as e:
        logger.error("Failed to load Whisper teacher: %s", e)
        raise

    saved = 0
    max_proc = min(len(ds), max_items) if hasattr(ds, '__len__') else max_items
    logger.info("Running Whisper precompute for up to %d items", max_proc)

    for i in range(max_proc):
        try:
            sample = ds[i]
        except Exception as e:
            logger.warning("Failed to index dataset at %d: %s", i, e)
            continue

        audio_field = sample.get('audio')
        wav = load_audio_debug(audio_field, i)
        if wav is None:
            logger.warning("No decoded audio for sample %d; skipping.", i)
            continue

        # Ensure numpy array float32 mono
        try:
            import numpy as _np
            arr = _np.array(wav, dtype=_np.float32)
            if arr.ndim > 1:
                # collapse multi-channel to mono
                arr = arr.mean(axis=1) if arr.shape[1] > 1 else arr.squeeze()
        except Exception as _e:
            logger.warning("Failed to normalize waveform for sample %d: %s", i, _e)
            continue

        try:
            # Processor -> input_features
            feats = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="pt")
            input_features = feats.get('input_features', None)
            if input_features is None:
                input_features = feats.get('input_values', None)
            if input_features is None:
                logger.warning("Processor returned no input features for sample %d", i)
                continue

            # Move inputs to device and match model parameter dtype (avoid float/half mismatch)
            try:
                model_dtype = next(whisper.parameters()).dtype
            except StopIteration:
                model_dtype = torch.float32
            input_features = input_features.to(device=device, dtype=model_dtype)
            with torch.no_grad():
                # Use the encoder directly to avoid needing decoder inputs
                try:
                    encoder_outputs = whisper.encoder(input_features=input_features)
                except Exception:
                    # fallback to calling model and letting it compute encoder+decoder
                    encoder_outputs = whisper.encoder(input_features)

            # encoder hidden states usually at encoder_outputs.last_hidden_state
            hidden = getattr(encoder_outputs, 'last_hidden_state', None)
            if hidden is None:
                # some versions return tuple-like outputs
                try:
                    hidden = encoder_outputs[0]
                except Exception:
                    hidden = None
            if hidden is None:
                logger.warning("Whisper produced no hidden states for sample %d", i)
                continue

            uid = _stable_uid(sample, i)
            save_path = os.path.join(out_dir, f"{uid}.pt")
            torch.save({
                "hidden_states": hidden.cpu(),
                "text": sample.get('text', ''),
                "audio_len": int(arr.shape[0]),
                "audio_path": (sample.get('audio', {}) or {}).get('path', '') if isinstance(sample.get('audio', {}), dict) else sample.get('audio', ''),
            }, save_path)
            saved += 1
            logger.info("Saved activation %d -> %s", i, save_path)
        except Exception as e:
            logger.exception("Error computing/saving activation for sample %d: %s", i, e)

    logger.warning("Whisper precompute complete, saved %d/%d samples to %s", saved, max_proc, out_dir)
