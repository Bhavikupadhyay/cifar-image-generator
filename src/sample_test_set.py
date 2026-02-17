import os
import torch
from torch.utils.data import DataLoader
from src import get_cifar_dataloaders, Config, save_images

def main():
    # Setup paths using Config for consistency
    cfg = Config(batch_size=64)
    output_dir = os.path.join(cfg.root_dir, 'assets')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data using project's existing dataloader logic
    print("Loading CIFAR-10 test set...")
    _, test_loader = get_cifar_dataloaders(cfg)

    # Get a batch
    images, _ = next(iter(test_loader))
    
    # Save as 4x16 grid (match generated results)
    # save_images already handles denormalization (inverse_transform) and grid creation
    print("Saving 4x16 grid to assets/real_samples.png...")
    out_path = os.path.join(output_dir, 'real_samples.png')
    save_images(images, out_path, nrow=4)
    
    print(f"Done! Saved to {out_path}")

if __name__ == '__main__':
    main()
