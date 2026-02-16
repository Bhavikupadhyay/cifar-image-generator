import torch
import torch.nn as nn
from src import Config, load_run_config, save_images
from src.modules import UNet, DDPM
import argparse
import os
import yaml

def get_config_from_run(run_name):
    # Locate the run directory
    # runs are stored in 'runs' folder by default as per Config
    # If run_name is a path, use it directly
    if os.path.exists(run_name):
        run_dir = run_name
    else:
        run_dir = os.path.join(os.getcwd(), 'runs', run_name)
    
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    config_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in: {run_dir}")

    # Load config dict
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Reconstruct Config object
    # We filter out keys that might not be in the current Config definition if schema changed, 
    # but ideally it should match.
    # For now, let's assume direct mapping.
    cfg = Config(**cfg_dict)
    
    # Overwrite run_dir to be absolute/correct for this machine if it was moved
    cfg.run_dir = run_dir
    cfg.ckpt_dir = os.path.join(run_dir, 'ckpt')
    cfg.samples_dir = os.path.join(run_dir, 'samples')
    cfg.log_dir = os.path.join(run_dir, 'logs')
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Inference for DDPM")
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run folder (timestamp) or path to it')
    parser.add_argument('--epoch', type=str, default='latest', help='Epoch to load (e.g. 5, 10, or "latest")')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of images to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--outfile', type=str, default='inference_sample.png', help='Output filename for the grid')
    
    args = parser.parse_args()
    
    # 1. Load Config
    cfg = get_config_from_run(args.run_name)
    print(f"Loaded config from: {cfg.run_dir}")
    print(f"Device: {args.device}")
    
    # 2. Init Models
    unet = UNet(
        n_channels=cfg.input_channels,
        n_classes=cfg.input_channels,
        time_emb_dim=cfg.embedding_dim,
        base_channels=cfg.base_channels
    ).to(args.device)

    ddpm = DDPM(
        unet=unet,
        n_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        embedding_dim=cfg.embedding_dim
    ).to(args.device)
    
    # 3. Load Checkpoint
    if args.epoch == 'latest':
        # Find latest checkpoint
        ckpts = [f for f in os.listdir(cfg.ckpt_dir) if f.endswith('.pth')]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {cfg.ckpt_dir}")
        # Sort by epoch number (assuming format vkpt_epoch_N.pth)
        # We can try to parse N
        def get_epoch(fname):
            try:
                return int(fname.split('_')[-1].split('.')[0])
            except:
                return 0
        latest_ckpt = sorted(ckpts, key=get_epoch)[-1]
        ckpt_path = os.path.join(cfg.ckpt_dir, latest_ckpt)
        print(f"Loading latest checkpoint: {latest_ckpt}")
    else:
        ckpt_path = os.path.join(cfg.ckpt_dir, f'ckpt_epoch_{args.epoch}.pth')
        print(f"Loading checkpoint: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    ddpm.load_state_dict(checkpoint)
    ddpm.eval()
    
    # 4. Inference
    print(f"Generating {args.num_samples} samples...")
    # Create inference output dir
    inference_dir = os.path.join(cfg.run_dir, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    
    with torch.no_grad():
        samples = ddpm.sample(args.num_samples, args.device)
        
    out_path = os.path.join(inference_dir, args.outfile)
    save_images(samples, out_path)
    print(f"Saved sample grid to: {out_path}")

if __name__ == '__main__':
    main()
