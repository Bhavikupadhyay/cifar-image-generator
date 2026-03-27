"""
INT8 Post-Training Quantization (PTQ) — one-time step per trained model.

Loads the latest checkpoint from a run, applies static INT8 quantization
to the UNet via torch.ao.quantization, calibrates on real CIFAR data,
and saves to <run_dir>/ckpt/quantized_int8.pth.

NOTE: torch.ao.quantization is CPU-only. This is intentional — quantized
models are for CPU-based serving (HuggingFace Spaces free tier).

Called by the CI pipeline:
    python quantize.py <run_name>
"""

import os
import sys
import torch
import torch.ao.quantization as tq

from src import Config, get_cifar_dataloaders, load_run_config, load_checkpoint
from src.modules import UNet, DDPM


class _QuantWrapper(torch.nn.Module):
    """Wraps UNet with quant/dequant stubs for static INT8 calibration."""
    def __init__(self, unet):
        super().__init__()
        self.quant = tq.QuantStub()
        self.unet = unet
        self.dequant = tq.DeQuantStub()

    def forward(self, x, t_emb):
        return self.dequant(self.unet(self.quant(x), t_emb))


def quantize(run_name, num_calib_batches=10):
    run_dir = run_name if os.path.exists(run_name) else os.path.join(os.getcwd(), 'runs', run_name)
    cfg = Config(**load_run_config(os.path.join(run_dir, 'config.yaml')))
    cfg.root_dir = os.getcwd()
    cfg.resume_run_folder(os.path.basename(run_dir))

    # Resolve latest checkpoint (skip any previously quantized file)
    ckpts = sorted(
        [f for f in os.listdir(cfg.ckpt_dir) if f.endswith('.pth') and 'quantized' not in f],
        key=lambda f: int(f.split('_')[-1].split('.')[0]) if f.split('_')[-1].split('.')[0].isdigit() else 0
    )
    ckpt_path = os.path.join(cfg.ckpt_dir, ckpts[-1])
    print(f'Loading: {ckpt_path}')

    # Detect norm type from checkpoint, then build model to match
    state, use_gn = load_checkpoint(ckpt_path, device='cpu')

    unet = UNet(
        n_channels=cfg.input_channels,
        n_classes=cfg.input_channels,
        time_emb_dim=cfg.embedding_dim,
        base_channels=cfg.base_channels,
        use_group_norm=use_gn,
    )
    ddpm = DDPM(
        unet=unet,
        n_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        embedding_dim=cfg.embedding_dim,
    )
    ddpm.load_state_dict(state)
    ddpm.eval()

    # Determine backend (x86 on Linux/Windows, qnnpack on macOS/ARM)
    import platform
    backend = 'qnnpack' if platform.system() == 'Darwin' else 'x86'
    torch.backends.quantized.engine = backend

    wrapped = _QuantWrapper(ddpm.unet)
    wrapped.eval()
    wrapped.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(wrapped, inplace=True)

    print(f'Calibrating on {num_calib_batches} batches (backend: {backend})...')
    train_loader, _ = get_cifar_dataloaders(cfg)
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= num_calib_batches:
                break
            t_emb = ddpm.time_embed(torch.randint(0, ddpm.n_timesteps, (x.shape[0],)))
            wrapped(x, t_emb)

    tq.convert(wrapped, inplace=True)
    ddpm.unet = wrapped.unet

    out_path = os.path.join(cfg.ckpt_dir, 'quantized_int8.pth')
    torch.save({'model_state_dict': ddpm.state_dict(), 'quantized': True}, out_path)
    print(f'Saved → {out_path}')
    return out_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python quantize.py <run_name>')
        sys.exit(1)
    quantize(sys.argv[1])
