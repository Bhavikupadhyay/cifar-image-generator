import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def get_cifar_dataloaders(cfg):
  full_data_path = os.path.join(cfg.root_dir, cfg.data_dir)

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  train_set = CIFAR10(root=full_data_path, train=True, download=True, transform=transform)
  test_set = CIFAR10(root=full_data_path, train=False, download=True, transform=transform)
  
  persistent = cfg.num_workers > 0

  train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    persistent_workers=persistent,
  )

  test_loader = DataLoader(
    test_set,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    persistent_workers=persistent,
  )

  return train_loader, test_loader