import logging

try:
    from datasets import load_dataset, Audio
    _HAVE_DATASETS = True
except Exception:
    _HAVE_DATASETS = False

logger = logging.getLogger(__name__)

def get_waxal_swahili(split="train", streaming=False):
    """
    Load the specific Swahili TTS subset from WAXAL.
    
    Args:
        split (str): 'train', 'validation', or 'test'.
        streaming (bool): If True, streams data (good for large datasets).
                          If False, downloads everything (good for caching).
    
    Returns:
        Dataset: The WAXAL Swahili dataset ready for training.
    """
    logger.info(f"Loading WAXAL (swa_tts) split={split}...")

    if not _HAVE_DATASETS:
        logger.warning("`datasets` library not available — returning synthetic generator for smoke runs.")

        def _synthetic_generator():
            i = 0
            while True:
                yield {
                    "text": f"synthetic sample {i}",
                    "audio": {"path": f"synthetic://sample/{i}"},
                    "id": i,
                }
                i += 1

        return _synthetic_generator()

    try:
        # Load specifically the 'swa_tts' config. Prefer non-streaming by default
        # so the dataset files are downloaded to disk (robust in remote runtimes).
        ds = load_dataset(
            "google/WaxalNLP",
            "swa_tts",
            split=split,
            streaming=streaming,
        )

        # Do NOT request automatic decoding here. Decoding in remote runtimes
        # (Modal) can silently return None for many formats. We prefer to leave
        # the dataset's `audio` column as metadata (paths/bytes) and decode
        # manually in `precompute.py` where we can surface detailed errors.
        return ds
    except Exception as exc:
        logger.error("Failed to load WAXAL: %s", exc)
        raise


def save_to_disk(ds, output_path: str):
    logger.info("Saving dataset to %s...", output_path)
    ds.save_to_disk(output_path)
    logger.info("Save complete.")
