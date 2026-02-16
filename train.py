import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataclasses import asdict
import os
import time

from src import Config, get_cifar_dataloaders, seed_everything, save_run_config, save_images
from src.modules import UNet, DDPM

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def _use_wandb(cfg):
    """Helper to check if W&B is active."""
    return cfg.use_wandb and HAS_WANDB


def _get_grad_norm(model):
    """Compute total L2 gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def train_ddpm(ddpm, train_loader, cfg: Config, device, writer=None):
    optimizer = Adam(ddpm.parameters(), lr=cfg.learning_rate)
    loss_history = []
    global_step = 0

    # Ensure samples dir exists
    os.makedirs(cfg.samples_dir, exist_ok=True)

    # Watch model gradients and parameters in W&B
    if _use_wandb(cfg):
        wandb.watch(ddpm, log='all', log_freq=100)

    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
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

            # Track gradient norm before optimizer step
            grad_norm = _get_grad_norm(ddpm)

            optimizer.step()
            global_step += 1

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            # --- Batch-level logging (every 50 steps to avoid overhead) ---
            if global_step % 50 == 0:
                if writer:
                    writer.add_scalar('Loss/batch', batch_loss, global_step)
                    writer.add_scalar('GradNorm/batch', grad_norm, global_step)
                if _use_wandb(cfg):
                    wandb.log({
                        "batch_loss": batch_loss,
                        "grad_norm": grad_norm,
                        "global_step": global_step,
                    })

        # --- Epoch-level metrics ---
        epoch_duration = time.time() - epoch_start
        epoch_avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_min_loss = min(batch_losses)
        epoch_max_loss = max(batch_losses)
        loss_history.append(epoch_avg_loss)

        # Log epoch metrics
        epoch_metrics = {
            "train_loss": epoch_avg_loss,
            "train_loss_min": epoch_min_loss,
            "train_loss_max": epoch_max_loss,
            "epoch": epoch + 1,
            "epoch_duration_sec": epoch_duration,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }

        # GPU memory tracking
        if device == 'cuda':
            epoch_metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            epoch_metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        if writer:
            writer.add_scalar('Loss/train', epoch_avg_loss, epoch + 1)
            writer.add_scalar('Loss/train_min', epoch_min_loss, epoch + 1)
            writer.add_scalar('Loss/train_max', epoch_max_loss, epoch + 1)
            writer.add_scalar('Timing/epoch_sec', epoch_duration, epoch + 1)
        if _use_wandb(cfg):
            wandb.log(epoch_metrics)

        # --- Sampling and Logging ---
        if (epoch + 1) % cfg.sample_every_epochs == 0 or (epoch + 1) == cfg.num_epochs:
            ddpm.eval()
            with torch.no_grad():
                samples = ddpm.sample(16, device)

            save_path = os.path.join(cfg.samples_dir, f'sample_{epoch+1}.png')
            save_images(samples, save_path)

            # Log sample images
            if writer:
                writer.add_images('Samples', samples, epoch + 1)
            if _use_wandb(cfg):
                wandb.log({
                    "samples": wandb.Image(save_path, caption=f"Epoch {epoch+1}"),
                    "epoch": epoch + 1,
                })

            # Revert to train mode
            ddpm.train()

        # --- Checkpointing ---
        if (epoch + 1) % cfg.save_every_epochs == 0 or (epoch + 1) == cfg.num_epochs:
            ckpt_path = os.path.join(cfg.ckpt_dir, f'ckpt_epoch_{epoch+1}.pth')
            torch.save(ddpm.state_dict(), ckpt_path)

            # Save checkpoint as W&B artifact for versioning
            if _use_wandb(cfg):
                artifact = wandb.Artifact(
                    f"model-ckpt-epoch-{epoch+1}",
                    type="model",
                    metadata={"epoch": epoch + 1, "avg_loss": epoch_avg_loss}
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)


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
        n_classes=cfg.input_channels,
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

    # Log model summary
    total_params = sum(p.numel() for p in ddpm.parameters())
    trainable_params = sum(p.numel() for p in ddpm.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')

    if _use_wandb(cfg):
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": device,
        })

    # Start Training
    train_ddpm(ddpm, train_loader, cfg, device, writer=writer)

    # Cleanup
    writer.close()
    if _use_wandb(cfg):
        wandb.finish()


if __name__ == '__main__':
    cfg = Config()
    train(cfg)