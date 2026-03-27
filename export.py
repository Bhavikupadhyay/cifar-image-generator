"""
ONNX export for the CIFAR DDPM denoising UNet.

Exports the UNet (with time embedding) to ONNX, then applies dynamic
INT8 quantization via ONNX Runtime. The sampling loop stays in Python;
the ONNX model handles only the per-step denoising forward pass.

Exported model inputs:
    x  : (batch, 3, 32, 32)  — noisy image at timestep t
    t  : (batch,)             — integer timestep [0, 999]

Exported model output:
    noise : (batch, 3, 32, 32) — predicted noise

Usage (called once per trained model, before building the Docker image):
    python export.py <run_name>

Outputs written to <run_dir>/export/:
    unet_fp32.onnx
    unet_int8.onnx   ← what the serving app loads
"""

import os
import sys
import torch
import torch.nn as nn

from src import Config, load_run_config, load_checkpoint
from src.modules import UNet, DDPM


class _DenoiserWrapper(nn.Module):
    """Bundles TimeEmbeddings + UNet so the ONNX model takes raw timestep ints."""
    def __init__(self, ddpm):
        super().__init__()
        self.time_embed = ddpm.time_embed
        self.unet = ddpm.unet

    def forward(self, x, t):
        return self.unet(x, self.time_embed(t))


def export(run_name, opset=17):
    run_dir = run_name if os.path.exists(run_name) else os.path.join(os.getcwd(), 'runs', run_name)
    cfg = Config(**load_run_config(os.path.join(run_dir, 'config.yaml')))
    cfg.root_dir = os.getcwd()
    cfg.resume_run_folder(os.path.basename(run_dir))

    # Resolve checkpoint — prefer INT8 pytorch file if available, else latest FP32
    ckpts = sorted(
        [f for f in os.listdir(cfg.ckpt_dir) if f.endswith('.pth') and 'quantized' not in f],
        key=lambda f: int(f.split('_')[-1].split('.')[0]) if f.split('_')[-1].split('.')[0].isdigit() else 0
    )
    ckpt_path = os.path.join(cfg.ckpt_dir, ckpts[-1])
    print(f'Loading: {ckpt_path}')

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

    export_dir = os.path.join(run_dir, 'export')
    os.makedirs(export_dir, exist_ok=True)

    # ── FP32 ONNX export ──────────────────────────────────────────────────────
    wrapper = _DenoiserWrapper(ddpm)
    wrapper.eval()

    dummy_x = torch.randn(1, cfg.input_channels, cfg.image_size, cfg.image_size)
    dummy_t = torch.zeros(1, dtype=torch.long)

    fp32_path = os.path.join(export_dir, 'unet_fp32.onnx')
    print(f'Exporting FP32 ONNX (opset {opset})...')
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_t),
        fp32_path,
        input_names=['x', 't'],
        output_names=['noise'],
        dynamic_axes={'x': {0: 'batch'}, 't': {0: 'batch'}, 'noise': {0: 'batch'}},
        opset_version=opset,
    )
    fp32_mb = os.path.getsize(fp32_path) / 1e6
    print(f'Saved {fp32_path}  ({fp32_mb:.1f} MB)')

    # ── INT8 dynamic quantization via ONNX Runtime ────────────────────────────
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(export_dir, 'unet_int8.onnx')
        print('Applying dynamic INT8 quantization...')
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
        int8_mb = os.path.getsize(int8_path) / 1e6
        print(f'Saved {int8_path}  ({int8_mb:.1f} MB)')
        print(f'Size reduction: {fp32_mb:.1f} MB → {int8_mb:.1f} MB  ({fp32_mb/int8_mb:.1f}x)')
    except ImportError:
        print('onnxruntime not installed — skipping INT8 step. Run: pip install onnxruntime')
        int8_path = None

    # ── Verify FP32 output matches PyTorch ────────────────────────────────────
    print('Verifying ONNX output vs PyTorch...')
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        pt_out = wrapper(dummy_x, dummy_t).numpy()
    ort_out = sess.run(['noise'], {'x': dummy_x.numpy(), 't': dummy_t.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f'Max absolute difference PyTorch vs ONNX: {max_diff:.2e}  {"✓" if max_diff < 1e-4 else "✗ LARGE — check model"}')

    return fp32_path, int8_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python export.py <run_name>')
        sys.exit(1)
    export(sys.argv[1])
