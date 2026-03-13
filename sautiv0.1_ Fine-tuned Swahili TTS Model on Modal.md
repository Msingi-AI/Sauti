# sautiv0.1: Fine-tuned Swahili TTS Model on Modal

This repository contains the necessary code and documentation to fine-tune a Swahili Text-to-Speech (TTS) model, `sautiv0.1`, using the Google Waxal dataset and deploy it on Modal.

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Project Structure](#2-project-structure)
3.  [Setup and Prerequisites](#3-setup-and-prerequisites)
4.  [Dataset](#4-dataset)
5.  [Fine-tuning the Model](#5-fine-tuning-the-model)
6.  [Inference](#6-inference)
7.  [Interactive Development with Jupyter Notebook](#7-interactive-development-with-jupyter-notebook)
8.  [Model Details](#8-model-details)
9.  [Acknowledgements](#9-acknowledgements)

## 1. Introduction

`sautiv0.1` is a Swahili TTS model fine-tuned from a powerful multilingual base model (XTTS v2) using the Google WaxalNLP dataset. The goal is to provide a high-quality, natural-sounding Swahili voice for various applications. The entire training and inference pipeline is designed to run efficiently on Modal, a serverless cloud computing platform, leveraging its GPU capabilities and persistent volumes.

## 2. Project Structure

```
. 
├── README.md
├── sautiv0.1_design.md
├── train_sauti.py
├── inference_sauti.py
└── notebook_sauti.py
```

*   `README.md`: This file, providing an overview and instructions.
*   `sautiv0.1_design.md`: Detailed design document outlining the architecture and data processing.
*   `train_sauti.py`: Python script for fine-tuning the XTTS v2 model on the Waxal Swahili TTS dataset using Modal.
*   `inference_sauti.py`: Python script for performing inference (generating speech) with the fine-tuned `sautiv0.1` model on Modal.
*   `notebook_sauti.py`: Python script to launch a Jupyter Lab instance on Modal for interactive development and experimentation.

## 3. Setup and Prerequisites

To run this project, you will need:

*   A Modal account and the Modal client installed (`pip install modal-client`).
*   `ffmpeg` installed on your system (for audio processing).
*   A Hugging Face account and an API token with write access if you intend to push the model to the Hugging Face Hub.

### Modal Volume Setup

A Modal persistent volume named `sauti-data` is used to store the dataset and model checkpoints. This volume needs to be created once:

```bash
modal volume create sauti-data
```

### Hugging Face Secret Setup

To push the fine-tuned model to the Hugging Face Hub, you need to create a Modal secret named `huggingface-secret` that contains your Hugging Face API token. This token should have write access.

```bash
modal secret create huggingface-secret --env-name HF_TOKEN
```

When prompted, paste your Hugging Face API token.

## 4. Dataset

The project utilizes the **Google WaxalNLP dataset**, specifically the `swa_tts` (Swahili Text-to-Speech) subset, available on Hugging Face [1]. This dataset provides high-quality audio recordings and corresponding text transcripts in Swahili, essential for fine-tuning the TTS model.

### Dataset Fields

The `swa_tts` subset includes the following fields:

| Field       | Description                                        |
| :---------- | :------------------------------------------------- |
| `id`        | Unique identifier for each sample.                 |
| `speaker_id`| Unique identifier for the speaker.                 |
| `audio`     | Audio data (includes `array` and `sampling_rate`). |
| `text`      | Text script corresponding to the audio.            |
| `locale`    | ISO 639-2 language code, set to `swa` for Swahili. |
| `gender`    | Speaker gender.                                    |

## 5. Fine-tuning the Model

The `train_sauti.py` script handles the entire fine-tuning process on Modal. It performs the following steps:

1.  **Environment Setup**: Sets up a Modal image with necessary dependencies (TTS library, datasets, torch, torchaudio, ffmpeg, huggingface_hub).
2.  **Dataset Loading and Preprocessing**: Downloads the `swa_tts` dataset, extracts audio, and creates a `metadata.csv` file in LJSpeech format within the Modal volume.
3.  **Model Configuration: Loads the pre-trained XTTS v2 model and configures it for full fine-tuning with Swahili data, including optimized training parameters for A100 GPUs such as increased epochs (20), larger batch size (16), and an adjusted learning rate (1e-5).
4.  **Training**: Initiates the fine-tuning process, saving checkpoints to the `sauti-data` Modal volume.
5.  **Push to Hugging Face Hub (Optional)**: If `hf_repo_id` is provided, the script will push the fine-tuned model to the specified Hugging Face repository.

To start the fine-tuning process, run the following command:

```bash
modal run train_sauti.py --hf-repo-id "your-username/sautiv0.1"
```

Replace `"your-username/sautiv0.1"` with your desired Hugging Face repository ID. If you omit the `--hf-repo-id` argument, the model will only be saved to the Modal volume.

This will launch a Modal job on 3x A100 GPU instances, leveraging Distributed Data Parallel (DDP) for efficient training, and store all processed data and model checkpoints in the `sauti-data` volume.

## 6. Inference

The `inference_sauti.py` script allows you to generate Swahili speech using the fine-tuned `sautiv0.1` model. It can load the model either from your local Modal volume or directly from the Hugging Face Hub.

To generate speech, run the script:

```bash
modal run inference_sauti.py --hf-repo-id "your-username/sautiv0.1"
```

Replace `"your-username/sautiv0.1"` with the Hugging Face repository ID where your model is stored. If you omit the `--hf-repo-id` argument, the script will attempt to load the model from the `sauti-data` Modal volume.

The `main` function in `inference_sauti.py` includes a sample text and saves the generated audio to `sautiv0.1_sample.wav`. You can modify this function to generate speech for different texts.

### Deploying as an API (Optional)

For a persistent inference endpoint, you can deploy the `generate_swahili_speech` function as a Modal API. Refer to Modal's documentation for deploying functions as web endpoints.

## 7. Interactive Development with Jupyter Notebook

For interactive development, experimentation, and debugging, you can launch a Jupyter Lab instance on Modal.

To start the Jupyter Lab server, run:

```bash
modal run notebook_sauti.py
```

Modal will provide a public URL in your terminal where you can access the Jupyter Lab interface. This notebook will have access to the `sauti-data` Modal volume, allowing you to interact with your dataset and fine-tuned models directly.

## 8. Model Details

*   **Model Name**: `sautiv0.1`
*   **Base Model**: XTTS v2 (from Coqui TTS)
*   **Language**: Swahili (`sw`)
*   **Dataset**: Google WaxalNLP (`swa_tts` subset)
*   **Training Platform**: Modal

## 9. Acknowledgements

*   **Google** for providing the WaxalNLP dataset.
*   **Coqui TTS** for the XTTS v2 model and their open-source TTS library.
*   **Modal** for providing the serverless GPU infrastructure for training and deployment.
*   **Hugging Face** for providing the platform for model sharing and collaboration.

## References

[1] google/WaxalNLP. (n.d.). *Datasets at Hugging Face*. Retrieved from [https://huggingface.co/datasets/google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP)
