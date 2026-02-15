import torch
from src import Config, save_run_config
from src.modules import UNet, DDPM
import os

def create_dummy():
    print("Creating dummy checkpoint for inference testing...")
    cfg = Config(
        run_name='inference_test_run',
        num_epochs=1,
        save_every_epochs=1,
        sample_every_epochs=1,
        batch_size=8,
        num_timesteps=10, 
        num_workers=0
    )
    
    # Create structure
    cfg.create_run_folder()
    save_run_config(cfg)
    
    # Init Models
    unet = UNet(
        n_channels=cfg.input_channels,
        n_classes=cfg.input_channels,
        time_emb_dim=cfg.embedding_dim,
        base_channels=cfg.base_channels
    )

    ddpm = DDPM(
        unet=unet,
        n_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        embedding_dim=cfg.embedding_dim
    )
    
    # Save Checkpoint
    ckpt_path = os.path.join(cfg.ckpt_dir, 'ckpt_epoch_1.pth')
    torch.save(ddpm.state_dict(), ckpt_path)
    print(f"Dummy checkpoint saved to: {ckpt_path}")

if __name__ == '__main__':
    create_dummy()
