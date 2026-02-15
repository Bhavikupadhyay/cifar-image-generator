# CIFAR Image Generator (DDPM)

A modern, modular PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) designed for generating high-quality images (specifically CIFAR-10). This repository serves as a clean, educational, and extensible codebase for understanding and experimenting with diffusion models.

## Key Features

-   **Modular Design**: The codebase is split into logical components (`UNet`, `DDPM`, `Dataset`, `Config`), making it easy to read and modify.
-   **Robust Configuration**: Powered by `dataclasses`, the `Config` system allows for type-safe and centralized hyperparameter management.
-   **Reproducibility**: Integrated seed setting and automatic experiment tracking (saving configs, checkpoints, and samples).
-   **Weights & Biases (WandB)**: Optional built-in integration for experiment logging and visualization.

## Architecture

This project implements the standard DDPM formulation:
-   **Backbone**: A customized U-Net (`src/modules/unet.py`) with:
    -   Time embeddings (Sinusoidal).
    -   Residual blocks (`DoubleConv`).
    -   Downsampling/Upsampling paths with skip connections.
    -   Self-attention mechanisms (optional/configurable).
-   **Diffusion Process**: Managed by the `DDPM` class (`src/modules/ddpm.py`), handling:
    -   Forward diffusion (adding noise).
    -   Reverse diffusion (denoising step-by-step).
    -   Noise schedule (linear beta schedule).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Bhavikupadhyay/cifar-image-generator.git
    cd cifar-image-generator
    ```

2.  **Set up the environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

### Training

To start a training run:
```bash
python train.py
```
**Configuration**:
You can modify hyperparameters in `src/config.py`. Key parameters include:
-   `num_epochs`: Total training epochs.
-   `batch_size`: Batch size for the dataloader.
-   `learning_rate`: learning rate for Adam optimizer.
-   `num_timesteps`: Number of diffusion steps (T).
-   `use_wandb`: Set to `True` for WandB logging.

All runs are saved to the `runs/` directory with a timestamp (e.g., `runs/2026-02-15_14-55-53`).

### Inference / Sampling

To generate images using a trained model:
```bash
# Basic usage (defaults to latest checkpoint in the run folder)
python inference.py --run_name <RUN_TIMESTAMP_OR_NAME>

# Advanced usage
python inference.py \
    --run_name 2026-02-15_14-55-53 \
    --epoch latest \
    --num_samples 64 \
    --device cuda \
    --outfile my_samples.png
```

## Project Structure

```
.
├── src/
│   ├── modules/
│   │   ├── unet.py         # U-Net architecture
│   │   ├── ddpm.py         # Diffusion logic (noise schedule, sampling)
│   │   └── blocks.py       # Neural network building blocks
│   ├── config.py           # Configuration dataclass
│   ├── dataset.py          # CIFAR-10 data loading
│   └── utils.py            # Helpers (save/load, seeding)
├── train.py                # Entry point for training
├── inference.py            # Entry point for generation
└── requirements.txt        # Python dependencies
```

## Extending the Project

-   **New Datasets**: Modify `src/dataset.py` to return a standard PyTorch DataLoader for your custom dataset.
-   **Architecture Tweaks**: Edit `src/modules/unet.py` to change the number of layers, channels, or add new attention mechanisms.
-   **Noise Schedules**: Modify the `beta_start` and `beta_end` in `src/config.py` or implement new schedules (cosine, sigmoid) in `src/modules/ddpm.py`.
