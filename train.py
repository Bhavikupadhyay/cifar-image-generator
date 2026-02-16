import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataclasses import asdict
import os

from src import Config, get_cifar_dataloaders, seed_everything, save_run_config, save_images
from src.modules import UNet, DDPM

try:
  import wandb
  HAS_WANDB = True
except ImportError:
  HAS_WANDB = False


def train_ddpm(ddpm, train_loader, cfg: Config, device, writer=None):
    optimizer = Adam(ddpm.parameters(), lr=cfg.learning_rate)
    loss_history = []

    # Ensure samples dir exists (it should be created by cfg.create_run_folder, but good to be safe if run standalone)
    os.makedirs(cfg.samples_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.num_epochs}')
        ddpm.train()
        
        batch_losses = []
        for batch in pbar:
            x, _ = batch
            x = x.to(device)

            optimizer.zero_grad()

            t = torch.randint(0, ddpm.n_timesteps, (x.shape[0],), device=device).long()
            loss = ddpm.reverse_losses(x, t)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_avg_loss = sum(batch_losses) / len(batch_losses)
        loss_history.append(epoch_avg_loss)

        # Log loss metrics
        if writer:
            writer.add_scalar('Loss/train', epoch_avg_loss, epoch + 1)
        if cfg.use_wandb and HAS_WANDB:
            wandb.log({"train_loss": epoch_avg_loss, "epoch": epoch + 1})

        # Sampling and Logging
        if (epoch + 1) % cfg.sample_every_epochs == 0 or (epoch + 1) == cfg.num_epochs:
            ddpm.eval()
            with torch.no_grad():
                samples = ddpm.sample(16, device) 
            
            save_path = os.path.join(cfg.samples_dir, f'sample_{epoch+1}.png')
            save_images(samples, save_path)

            # Log sample images
            if writer:
                writer.add_images('Samples', samples, epoch + 1)
            if cfg.use_wandb and HAS_WANDB:
                wandb.log({"samples": wandb.Image(save_path), "epoch": epoch + 1})
            
            # Revert to train mode
            ddpm.train()
        
        # Checkpointing
        if (epoch + 1) % cfg.save_every_epochs == 0 or (epoch + 1) == cfg.num_epochs:
            ckpt_path = os.path.join(cfg.ckpt_dir, f'ckpt_epoch_{epoch+1}.pth')
            torch.save(ddpm.state_dict(), ckpt_path)


def train(cfg: Config, resume_id=None):
    if resume_id:
        cfg.resume_run_folder(resume_id)
    else:
        cfg.create_run_folder()
        save_run_config(cfg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(cfg.seed)

    print(f'Starting Run: {cfg.run_name}')
    print(f'Device: {device}')
    print(f'Logs: {cfg.log_dir}')

    writer = SummaryWriter(log_dir=cfg.log_dir)

    if cfg.use_wandb and HAS_WANDB:
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=asdict(cfg), entity=cfg.wandb_entity)

    train_loader, _ = get_cifar_dataloaders(cfg)

    # Initialize Models
    unet = UNet(
        n_channels=cfg.input_channels,
        n_classes=cfg.input_channels, # For unconditional generation in this setup, or as defined in unet
        time_emb_dim=cfg.embedding_dim,
        base_channels=cfg.base_channels
    ).to(device)

    ddpm = DDPM(
        unet=unet,
        n_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        embedding_dim=cfg.embedding_dim
    ).to(device)

    # Start Training
    train_ddpm(ddpm, train_loader, cfg, device, writer=writer)

    # Cleanup
    writer.close()
    if cfg.use_wandb and HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    cfg = Config()
    train(cfg)