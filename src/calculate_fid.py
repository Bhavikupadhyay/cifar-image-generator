import torch
import os
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from src import Config, get_cifar_dataloaders, load_run_config, inverse_transform
from src.modules import UNet, DDPM

def denormalize_to_uint8(tensors):
    """
    Helper to convert tensors from [-1, 1] to [0, 255] uint8.
    Uses inverse_transform from utils.py.
    """
    return (inverse_transform(tensors) * 255).add_(0.5).clamp(0, 255).to(torch.uint8)

def calculate_fid(run_name, num_samples=1000, batch_size=64, device=None):
    """
    Calculates FID score for a specific run.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load Config (DRY: Using Config and load_run_config)
    run_dir = os.path.join(os.getcwd(), 'runs', run_name) if not os.path.exists(run_name) else run_name
    cfg_path = os.path.join(run_dir, 'config.yaml')
    
    cfg = Config(**load_run_config(cfg_path))
    # Override root_dir for local execution (handles runs from Colab)
    cfg.root_dir = os.getcwd()
    cfg.resume_run_folder(run_name)

    # 2. Init Models
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

    # 3. Load Latest Checkpoint
    ckpts = [f for f in os.listdir(cfg.ckpt_dir) if f.endswith('.pth')]
    def get_epoch(f):
        try: return int(f.split('_')[-1].split('.')[0])
        except: return 0
    latest_ckpt = sorted(ckpts, key=get_epoch)[-1]
    ckpt_path = os.path.join(cfg.ckpt_dir, latest_ckpt)
    
    print(f"Loading checkpoint: {ckpt_path}")
    ddpm.load_state_dict(torch.load(ckpt_path, map_location=device))
    ddpm.eval()

    # 4. FID Metric
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # 5. Process Real Images (DRY: Using get_cifar_dataloaders)
    train_loader, test_loader = get_cifar_dataloaders(cfg)
    print(f"Collecting {num_samples} real images...")
    processed = 0
    for images, _ in test_loader:
        imgs = denormalize_to_uint8(images).to(device)
        fid_metric.update(imgs, real=True)
        processed += imgs.shape[0]
        if processed >= num_samples: break

    # 6. Generate Fake Images
    print(f"Generating {num_samples} fake images...")
    processed = 0
    while processed < num_samples:
        curr_batch = min(batch_size, num_samples - processed)
        samples = ddpm.sample(curr_batch, device)
        imgs = denormalize_to_uint8(samples).to(device)
        fid_metric.update(imgs, real=False)
        processed += curr_batch

    score = fid_metric.compute().item()
    print(f"FID Score: {score:.4f}")
    return score

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, help='Name or path of the run')
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()
    
    calculate_fid(args.run_name, num_samples=args.num_samples)
