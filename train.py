import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataclasses import asdict
import os
import time
import copy

from src import Config, get_cifar_dataloaders, seed_everything, save_run_config, save_images
from src.modules import UNet, DDPM

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from src.calculate_fid import compute_fid
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


def _use_wandb(cfg):
    return cfg.use_wandb and HAS_WANDB


def _get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v

    def state_dict(self):
        return self.shadow


def _build_optimizer(ddpm, cfg, device):
    try:
        return torch.optim.AdamW(
            ddpm.parameters(), lr=cfg.learning_rate, fused=(device == 'cuda')
        )
    except TypeError:
        return torch.optim.AdamW(ddpm.parameters(), lr=cfg.learning_rate)


def train_ddpm(ddpm, train_loader, cfg: Config, device, writer=None):
    optimizer = _build_optimizer(ddpm, cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == 'cuda'))
    ema = EMA(ddpm, decay=cfg.ema_decay) if cfg.use_ema else None

    loss_history = []
    global_step = 0

    os.makedirs(cfg.samples_dir, exist_ok=True)

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

            with torch.autocast(device_type=device, enabled=(cfg.use_amp and device == 'cuda')):
                loss = ddpm.reverse_losses(x, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)   # unscale before grad norm so the value is meaningful
            grad_norm = _get_grad_norm(ddpm)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(ddpm)

            global_step += 1

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

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

        epoch_duration = time.time() - epoch_start
        epoch_avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_min_loss = min(batch_losses)
        epoch_max_loss = max(batch_losses)
        loss_history.append(epoch_avg_loss)

        epoch_metrics = {
            "train_loss": epoch_avg_loss,
            "train_loss_min": epoch_min_loss,
            "train_loss_max": epoch_max_loss,
            "epoch": epoch + 1,
            "epoch_duration_sec": epoch_duration,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }

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

        if (epoch + 1) % cfg.sample_every_epochs == 0 or (epoch + 1) == cfg.num_epochs:
            ddpm.eval()
            with torch.no_grad():
                samples = ddpm.sample(16, device)

            save_path = os.path.join(cfg.samples_dir, f'sample_{epoch+1}.png')
            save_images(samples, save_path)

            if writer:
                writer.add_images('Samples', samples, epoch + 1)
            if _use_wandb(cfg):
                wandb.log({
                    "samples": wandb.Image(save_path, caption=f"Epoch {epoch+1}"),
                    "epoch": epoch + 1,
                })

            # --- Intermediate FID ---
            if (HAS_TORCHMETRICS
                    and cfg.fid_every_epochs > 0
                    and (epoch + 1) % cfg.fid_every_epochs == 0):
                print(f'Computing FID ({cfg.fid_num_samples} samples)...')
                fid_score = compute_fid(ddpm, cfg, device, num_samples=cfg.fid_num_samples)
                print(f'FID @ epoch {epoch+1}: {fid_score:.2f}')
                if writer:
                    writer.add_scalar('FID/train', fid_score, epoch + 1)
                if _use_wandb(cfg):
                    wandb.log({"fid": fid_score, "epoch": epoch + 1})

            ddpm.train()

        if (epoch + 1) % cfg.save_every_epochs == 0 or (epoch + 1) == cfg.num_epochs:
            ckpt_path = os.path.join(cfg.ckpt_dir, f'ckpt_epoch_{epoch+1}.pth')
            payload = {"model_state_dict": ddpm.state_dict()}
            if ema is not None:
                payload["ema_state_dict"] = ema.state_dict()
            torch.save(payload, ckpt_path)

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
    print(f'AMP: {cfg.use_amp}  EMA: {cfg.use_ema}  Compile: {cfg.use_compile}')
    print(f'Logs: {cfg.log_dir}')

    writer = SummaryWriter(log_dir=cfg.log_dir)

    if cfg.use_wandb and HAS_WANDB:
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=asdict(cfg), entity=cfg.wandb_entity)

    train_loader, _ = get_cifar_dataloaders(cfg)

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

    if cfg.use_compile:
        ddpm = torch.compile(ddpm)

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

    train_ddpm(ddpm, train_loader, cfg, device, writer=writer)

    writer.close()
    if _use_wandb(cfg):
        wandb.finish()


if __name__ == '__main__':
    cfg = Config()
    train(cfg)
