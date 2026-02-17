import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def main():
    # Setup paths
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'assets')
    os.makedirs(output_dir, exist_ok=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load test set
    print("Loading CIFAR-10 test set...")
    test_set = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    # Get a batch
    images, _ = next(iter(test_loader))
    
    # Save as 4x16 grid (match output.png)
    print("Saving 4x16 grid to samples/test_samples.png...")
    
    # Denormalize
    images = (images.clamp(-1, 1) + 1.0) / 2.0
    
    grid = make_grid(images, nrow=4)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255)
    grid = grid.permute(1, 2, 0).to('cpu', torch.uint8)
    ndarr = grid.numpy()

    plt.imsave(os.path.join(output_dir, 'real_samples.png'), ndarr)
    print("Done!")

if __name__ == '__main__':
    main()
