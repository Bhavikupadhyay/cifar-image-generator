import os
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Config:
  # --- Project Root Params ---
  project_name: str = 'cifar_ddpm'
  root_dir: str = os.getcwd()
  data_dir: str = 'data'
  experiments_dir: str = 'runs'

  # --- Training Hyperparameters ---
  batch_size: int = 256
  num_workers: int = 2
  num_epochs: int = 100
  learning_rate: float = 2e-4
  seed: int = 42

  # --- Checkpointing and Logging ---
  save_every_epochs: int = 10
  sample_every_epochs: int = 10
  use_wandb: bool = False
  wandb_entity: str = None

  # --- Diffusion Parameters ---
  num_timesteps: int = 1000
  beta_start: float = 1e-4
  beta_end: float = 0.02
  image_size: int = 32
  input_channels: int = 3

  # --- Model Architecture ---
  embedding_dim: int = 256
  base_channels: int = 128


  # --- Dynamic Paths for a specific run ---
  run_name: str = None
  run_dir: str = None
  ckpt_dir: str = None
  samples_dir: str = None
  log_dir: str = None


  def create_run_folder(self):
    """ Create a NEW Timestamped run """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    self.run_name = timestamp
    self._setup_directories()

    print(f'Directory Structure created at: {self.run_dir}')

  def resume_run_folder(self, run_name):
    """ Point to an EXISTING run folder """
    self.run_name = run_name
    self._setup_directories()

    print(f'Resuming run from: {self.run_dir}')

  def _setup_directories(self):
    """ Helper to set paths based on run_name """
    self.run_dir = os.path.join(self.root_dir, self.experiments_dir, self.run_name)
    self.ckpt_dir = os.path.join(self.run_dir, 'ckpt')
    self.samples_dir = os.path.join(self.run_dir, 'samples')
    self.log_dir = os.path.join(self.run_dir, 'logs')

    os.makedirs(self.ckpt_dir, exist_ok=True)
    os.makedirs(self.samples_dir, exist_ok=True)
    os.makedirs(self.log_dir, exist_ok=True)