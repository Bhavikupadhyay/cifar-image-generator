# CIFAR Image Generator (DDPM)

A modern implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating CIFAR-10 like images, built with PyTorch.

## Features
- **Modular Architecture**: Clean separation of model components (`UNet`, `DDPM`), configuration, and data loading.
- **Configurable**: Uses a robust `Config` system for easy hyperparameter tuning.
- **Inference Ready**: Includes a dedicated inference script for generating samples from trained models.
- **Reproducible**: Seeded runs and organized experiment tracking.

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
To train the model from scratch:
```bash
python train.py
```
This will:
-   Download the CIFAR-10 dataset (if not present).
-   Create a new experiment folder in `runs/<timestamp>`.
-   Save checkpoints and sample images periodically.

### Inference
 To generate images using a trained model:
```bash
# Load the latest checkpoint from a specific run
python inference.py --run_name <RUN_TIMESTAMP> --num_samples 16

# Example
python inference.py --run_name 2026-02-15_14-55-53 --device cpu
```

## Project Structure
```
.
├── src/                # Source code
│   ├── modules/        # Neural network modules (UNet, DDPM)
│   ├── config.py       # Configuration management
│   ├── dataset.py      # Data loading
│   └── utils.py        # Utilities
├── train.py            # Main training script
├── inference.py        # Inference script
└── requirements.txt    # Dependencies
```

## Branches
-   `main`: Stable development branch.
-   `feature/verify-pipeline`: Contains additional scripts for full pipeline verification.
-   `cleanup/prepare-public`: Cleaned up version ready for public release (no temp scripts).
