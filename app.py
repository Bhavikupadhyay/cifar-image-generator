"""
FastAPI serving app for the CIFAR DDPM generator.

Loads the INT8 ONNX model at startup. The DDPM reverse diffusion loop
runs in pure numpy — no PyTorch needed at serve time.

Environment variables:
    MODEL_PATH    path to unet_int8.onnx  (default: model/unet_int8.onnx)
    MODEL_CONFIG  path to config.yaml     (default: model/config.yaml)

Endpoints:
    GET  /health
    POST /generate?num_images=1
"""

import io
import os
import yaml
import numpy as np
import onnxruntime as ort
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse


# ── Diffusion schedule (rebuilt from config, no PyTorch needed) ───────────────

class _Schedule:
    def __init__(self, n_timesteps, beta_start, beta_end):
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

        self.n_timesteps = n_timesteps
        self.betas = betas
        self.sqrt_recip_alphas = np.sqrt(1.0 / alphas).astype(np.float32)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod).astype(np.float32)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).astype(np.float32)


# ── App state ─────────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "model/unet_int8.onnx")
    config_path = os.getenv("MODEL_CONFIG", "model/config.yaml")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path!r}. Run export.py first.")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config not found at {config_path!r}.")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _state["session"] = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    _state["schedule"] = _Schedule(
        n_timesteps=cfg.get("num_timesteps", 1000),
        beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02),
    )
    _state["cfg"] = cfg
    print(f"Loaded: {model_path}")
    yield
    _state.clear()


app = FastAPI(title="CIFAR DDPM Generator", lifespan=lifespan)


# ── Sampling logic ────────────────────────────────────────────────────────────

def _denoise_step(x: np.ndarray, t_idx: int) -> np.ndarray:
    """Single reverse diffusion step via ONNX inference."""
    session: ort.InferenceSession = _state["session"]
    sched: _Schedule = _state["schedule"]

    t = np.full((x.shape[0],), t_idx, dtype=np.int64)
    predicted_noise = session.run(["noise"], {"x": x, "t": t})[0]

    mean = sched.sqrt_recip_alphas[t_idx] * (
        x - sched.betas[t_idx] * predicted_noise / sched.sqrt_one_minus_alphas_cumprod[t_idx]
    )

    if t_idx == 0:
        return mean

    z = np.random.randn(*x.shape).astype(np.float32)
    return mean + np.sqrt(sched.posterior_variance[t_idx]) * z


def _sample(num_images: int) -> np.ndarray:
    """Full reverse diffusion: pure Gaussian noise → denoised images."""
    sched: _Schedule = _state["schedule"]
    cfg = _state["cfg"]

    x = np.random.randn(
        num_images,
        cfg.get("input_channels", 3),
        cfg.get("image_size", 32),
        cfg.get("image_size", 32),
    ).astype(np.float32)

    for t_idx in reversed(range(sched.n_timesteps)):
        x = _denoise_step(x, t_idx)

    return np.clip(x, -1.0, 1.0)


def _to_png_grid(images: np.ndarray, nrow: int = 4) -> bytes:
    """Convert (N, C, H, W) float32 array in [-1, 1] to a PNG grid."""
    imgs = ((images + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    n, c, h, w = imgs.shape
    ncols = min(n, nrow)
    nrows = (n + ncols - 1) // ncols

    grid = np.zeros((nrows * h, ncols * w, c), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, col = divmod(i, ncols)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = img.transpose(1, 2, 0)

    buf = io.BytesIO()
    Image.fromarray(grid).save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "session" in _state}


@app.post("/generate")
def generate(num_images: int = 1):
    if not 1 <= num_images <= 4:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 4")

    samples = _sample(num_images)
    return StreamingResponse(_to_png_grid(samples), media_type="image/png")
