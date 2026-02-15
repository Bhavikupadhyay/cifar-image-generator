import torch
from src import Config
from train import train
import os

def bootstrap():
    print("Bootstrapping a run for inference testing...")
    # Small run to generate checkpoint
    cfg = Config(
        run_name='inference_test_run',
        num_epochs=1,
        save_every_epochs=1,
        sample_every_epochs=1,
        batch_size=8,
        num_timesteps=10, 
        num_workers=0,
        use_wandb=False
    )
    
    # Run training
    train(cfg)
    print("Bootstrap run complete.")

if __name__ == '__main__':
    bootstrap()
