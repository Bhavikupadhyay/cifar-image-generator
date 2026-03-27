"""
FastAPI + Gradio serving app for the CIFAR DDPM generator.

Loads the INT8 ONNX model at startup. Uses DDIM sampling (50 steps, eta=0)
for fast deterministic generation — no PyTorch needed at serve time.

Environment variables:
    MODEL_PATH      path to unet_int8.onnx  (default: model/unet_int8.onnx)
    MODEL_CONFIG    path to config.yaml     (default: model/config.yaml)
    DDIM_STEPS      number of DDIM steps    (default: 50)

Endpoints:
    GET  /health
    POST /generate?num_images=1
    GET  /           (Gradio interface)
"""

import io
import os
import yaml
import numpy as np
import onnxruntime as ort
import gradio as gr
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse


# ── Diffusion schedule ────────────────────────────────────────────────────────

class _Schedule:
    def __init__(self, n_timesteps, beta_start, beta_end):
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)
        alphas = 1.0 - betas
        self.n_timesteps = n_timesteps
        self.alphas_cumprod = np.cumprod(alphas).astype(np.float32)


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
    _state["ddim_steps"] = int(os.getenv("DDIM_STEPS", 50))
    print(f"Loaded: {model_path}  |  DDIM steps: {_state['ddim_steps']}")
    yield
    _state.clear()


app = FastAPI(title="CIFAR DDPM Generator", lifespan=lifespan)


# ── DDIM sampling ─────────────────────────────────────────────────────────────

def _sample(num_images: int) -> np.ndarray:
    """DDIM reverse diffusion (eta=0, deterministic) in num_steps steps."""
    sched: _Schedule = _state["schedule"]
    cfg = _state["cfg"]
    session: ort.InferenceSession = _state["session"]
    num_steps: int = _state["ddim_steps"]

    # Evenly spaced timestep indices from T-1 down to 0
    T = sched.n_timesteps
    step_indices = np.linspace(0, T - 1, num_steps + 1, dtype=int)[::-1]

    x = np.random.randn(
        num_images,
        cfg.get("input_channels", 3),
        cfg.get("image_size", 32),
        cfg.get("image_size", 32),
    ).astype(np.float32)

    for i in range(num_steps):
        t_idx = int(step_indices[i])
        t_prev_idx = int(step_indices[i + 1])

        t = np.full((num_images,), t_idx, dtype=np.int64)
        eps = session.run(["noise"], {"x": x, "t": t})[0]

        alpha_t = sched.alphas_cumprod[t_idx]
        alpha_t_prev = sched.alphas_cumprod[t_prev_idx] if t_prev_idx > 0 else 1.0

        # Predicted x0
        x0_pred = (x - np.sqrt(1.0 - alpha_t) * eps) / np.sqrt(alpha_t)
        x0_pred = np.clip(x0_pred, -1.0, 1.0)

        # DDIM update (eta=0 → no stochastic term)
        x = np.sqrt(alpha_t_prev) * x0_pred + np.sqrt(1.0 - alpha_t_prev) * eps

    return np.clip(x, -1.0, 1.0)


# ── Image utilities ───────────────────────────────────────────────────────────

UPSCALE = 8  # 32px → 256px per image

def _to_pil_grid(images: np.ndarray, nrow: int = 4) -> Image.Image:
    """Convert (N, C, H, W) float32 in [-1,1] to an upscaled PIL grid."""
    imgs = ((images + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    n, c, h, w = imgs.shape
    ncols = min(n, nrow)
    nrows = (n + ncols - 1) // ncols

    grid = np.zeros((nrows * h, ncols * w, c), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, col = divmod(i, ncols)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = img.transpose(1, 2, 0)

    pil = Image.fromarray(grid)
    return pil.resize((pil.width * UPSCALE, pil.height * UPSCALE), Image.NEAREST)


def _to_png_bytes(images: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    _to_pil_grid(images).save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "session" in _state}


@app.post("/generate")
def generate(num_images: int = 1):
    if not 1 <= num_images <= 4:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 4")
    return StreamingResponse(_to_png_bytes(_sample(num_images)), media_type="image/png")


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def _gradio_generate(num_images: int) -> Image.Image:
    return _to_pil_grid(_sample(int(num_images)))


with gr.Blocks(title="CIFAR-10 Diffusion Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# CIFAR-10 Diffusion Model\n"
        "Generates images via DDIM sampling (50 steps). "
        "Each run produces fresh random samples."
    )
    num_slider = gr.Slider(minimum=1, maximum=4, step=1, value=4, label="Number of images")
    btn = gr.Button("Generate", variant="primary", size="lg")
    output = gr.Image(label="Generated images", type="pil", show_download_button=True)
    btn.click(fn=_gradio_generate, inputs=[num_slider], outputs=[output])


app = gr.mount_gradio_app(app, demo, path="/")
