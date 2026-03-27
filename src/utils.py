import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dataclasses import asdict

def seed_everything(seed=42):
  """ Sets a seed for reproducibility """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False # Slower but deterministic

def save_run_config(cfg):
  """ Saves the Config object as a YAML file in the run directory """
  if cfg.run_dir is None:
    raise ValueError('Run directory not set. Call cfg.create_run_folder() first')

  config_path = os.path.join(cfg.run_dir, 'config.yaml')

  cfg_dict = asdict(cfg)

  with open(config_path, 'w') as f:
    yaml.dump(cfg_dict, f, default_flow_style=False)

  print(f'Config saved to {config_path}')

def load_run_config(yaml_path):
  """ Loads a YAML file back into a dict (useful for inference and resuming training) """
  with open(yaml_path, 'r') as f:
    return yaml.safe_load(f)

def detect_norm_type(state_dict):
  """Return True if checkpoint was trained with GroupNorm, False for BatchNorm."""
  return any('gn1' in k for k in state_dict.keys())

def load_checkpoint(ckpt_path, device='cpu'):
  """Load a checkpoint and return (state_dict, use_group_norm)."""
  raw = torch.load(ckpt_path, map_location=device)
  if isinstance(raw, dict) and 'model_state_dict' in raw:
    state = raw.get('ema_state_dict') or raw['model_state_dict']
  else:
    state = raw
  return state, detect_norm_type(state)

def inverse_transform(tensors):
  """ Convert tensors from [-1, 1] back to [0, 1] for plotting """
  return (tensors.clamp(-1, 1) + 1.0) / 2.0

def save_images(images, path, nrow=4):
  """ Saves a batch of images to the specified path """
  if images.min() < 0:
    images = inverse_transform(images)

  grid = make_grid(images, nrow=nrow)

  grid = grid.mul(255).add_(0.5).clamp_(0, 255)
  grid = grid.permute(1, 2, 0).to('cpu', torch.uint8)
  ndarr = grid.numpy()

  plt.imsave(path, ndarr)