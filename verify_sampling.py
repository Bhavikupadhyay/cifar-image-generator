import torch
from src import Config, seed_everything
from src.modules import UNet, DDPM
import os

def verify_sampling():
    print("Verifying Sampling...")
    cfg = Config()
    device = 'cpu'
    
    # Init Models
    unet = UNet(
        n_channels=cfg.input_channels,
        n_classes=cfg.input_channels,
        time_emb_dim=cfg.embedding_dim,
        base_channels=cfg.base_channels
    ).to(device)

    ddpm = DDPM(
        unet=unet,
        n_timesteps=10, # usage small timesteps for speed
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        embedding_dim=cfg.embedding_dim
    ).to(device)
    
    ddpm.eval()
    
    try:
        with torch.no_grad():
            print("Starting sample generation...")
            samples = ddpm.sample(2, device) # Generate just 2 samples
            print(f"Sample shape: {samples.shape}")
            assert samples.shape == (2, 3, 32, 32)
            print("Sampling verification SUCCESS!")
    except Exception as e:
        print(f"Sampling verification FAILED: {e}")
        raise e

if __name__ == '__main__':
    verify_sampling()
