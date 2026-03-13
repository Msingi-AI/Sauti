# sautiv0.1 Fine-tuning Design Document

## 1. Introduction

This document outlines the design for fine-tuning a Swahili Text-to-Speech (TTS) model, named **sautiv0.1**, using the Google Waxal dataset. The fine-tuning process will leverage Modal for GPU-accelerated training and persistent storage.

## 2. Base TTS Model Selection: XTTS v2

### Rationale

XTTS v2 (from Coqui TTS) has been selected as the base model due to its robust performance, multilingual capabilities, and the availability of fine-tuning resources and community support. It is known for producing high-quality, natural-sounding speech and is well-suited for adaptation to new languages with sufficient data.

### Compatibility with Swahili

While XTTS v2 has strong multilingual support, explicit confirmation of its performance on Swahili will be part of the evaluation phase. However, its architecture is designed to generalize across languages, making it a suitable candidate.

## 3. Fine-tuning Architecture Overview

Fine-tuning XTTS v2 typically involves adapting the pre-trained model's weights to a new dataset. The core components of the XTTS v2 architecture include:

- **Encoder**: Processes the input text into a sequence of hidden representations.
- **Decoder**: Generates mel-spectrograms from the encoder's output.
- **Vocoder**: Converts mel-spectrograms into raw audio waveforms.

During fine-tuning, the entire model or specific layers (for example, language-specific layers in the encoder) are updated using the Swahili data. The goal is to retain the general speech synthesis capabilities of the pre-trained model while learning the specific phonetic and prosodic characteristics of Swahili from the Waxal dataset.

## 4. Dataset Processing Pipeline

### 4.1 Dataset Source: Google WaxalNLP (`swa_tts` subset)

The `google/WaxalNLP` dataset on Hugging Face provides a dedicated `swa_tts` subset for Swahili Text-to-Speech. This dataset contains:

- `id`: Unique identifier.
- `speaker_id`: Unique identifier for the speaker.
- `audio`: Audio data (including `array` for decoded audio bytes and `sampling_rate`).
- `text`: Text script corresponding to the audio.
- `locale`: ISO 639-2 language code (`swa`).
- `gender`: Speaker gender.

### 4.2 Preprocessing Steps

1. **Loading Data**: The `datasets` library will be used to load the `swa_tts` subset from Hugging Face.

```python
from datasets import load_dataset

tts_data = load_dataset("google/WaxalNLP", "swa_tts")
train_data = tts_data["train"]
# Potentially use validation and test splits if available and necessary for fine-tuning
```

2. **Audio Resampling**: XTTS v2 typically operates at a specific sampling rate (for example, 22050 Hz or 24000 Hz). The Waxal dataset audio will need to be resampled to match the model's expected input sampling rate if they differ. This will be done using `torchaudio.transforms.Resample` or a similar approach.

3. **Text Normalization**: The input text (`text` field) will undergo normalization to ensure consistency and improve synthesis quality. This may include:

- Lowercasing.
- Removal of special characters or punctuation not relevant for speech synthesis.
- Expansion of abbreviations or numbers (for example, `123` to `one hundred twenty-three`).
- Phonemization (conversion of text to phonetic representations), if the XTTS v2 fine-tuning process requires it or if it significantly improves Swahili pronunciation.

4. **Audio Length Filtering**: Training data often benefits from filtering out very short or very long audio samples to prevent training instabilities or memory issues. A reasonable range for audio duration will be determined and applied.

5. **Data Formatting**: The preprocessed audio and text data will be formatted into a structure compatible with the XTTS v2 fine-tuning script, likely involving `torch.Tensor` for audio and tokenized sequences for text.

## 5. Modal Training Environment Considerations

- **GPU Utilization**: Modal GPU capabilities will be utilized for efficient training. The training script will be designed to maximize GPU usage.
- **Persistent Volumes**: A Modal Volume will be used to store model checkpoints, logs, and potentially preprocessed datasets to enable resumable training and prevent data loss across runs.
- **Dependency Management**: All necessary Python packages (for example, `datasets`, `torchaudio`, and the Coqui `TTS` library) will be specified in the Modal environment setup.

## 6. Next Steps

Proceed to implement the dataset processing pipeline and the Modal training script based on this design.
