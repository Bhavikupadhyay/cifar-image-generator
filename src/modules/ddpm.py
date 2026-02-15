import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .unet import TimeEmbeddings, UNet


class DDPM(nn.Module):
    def __init__(self, unet, n_timesteps=1000, beta_start=1e-4, beta_end=2e-2, embedding_dim=256):
        super().__init__()

        self.unet = unet
        self.n_timesteps = n_timesteps
        self.embedding_dim = embedding_dim

        # Time Embeddings
        self.time_embed = TimeEmbeddings(embedding_dim=embedding_dim, max_time=n_timesteps)

        # Beta Schedule
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, n_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        # Setting up the calculations for Forward Diffusion (q)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # Setting up the calculations for Posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))

        posterior_variance = self.betas * (1.0 - torch.cat([torch.tensor([1.0]).to(self.betas.device), self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)


    def extract(self, a, t, x_shape):
      batch_size = t.shape[0]
      out = a.to(t.device)[t]
      return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_sample(self, x_0, t, noise=None):
      if noise is None:
        noise = torch.randn_like(x_0)

      sqrt_alpha_bar_t = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
      sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

      return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise


    def reverse_losses(self, x_0, t):
      noise = torch.randn_like(x_0).to(x_0.device)

      # Get noisy image
      x_noisy = self.forward_sample(x_0, t, noise=noise)

      # Get time embedding vector
      t_emb = self.time_embed(t)

      # Predict noise using UNet
      predicted_noise = self.unet(x_noisy, t_emb)

      return F.mse_loss(noise, predicted_noise)


    @torch.no_grad()
    def reverse_sample(self, x, t, t_index):
      betas_t = self.extract(self.betas, t, x.shape)
      sqrt_one_minus_alpha_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
      sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

      t_emb = self.time_embed(t)

      model_mean = sqrt_recip_alphas_t * (
          x - (betas_t * self.unet(x, t_emb)) / sqrt_one_minus_alpha_cumprod_t
      )

      if t_index == 0:
        return model_mean

      posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
      noise = torch.randn_like(x)

      return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def sample(self, batch_size, device, return_all_steps=False):
      img = torch.randn(batch_size, 3, 32, 32).to(device)

      imgs = []
      if return_all_steps:
        imgs.append(img)

      for i in tqdm(reversed(range(1, self.n_timesteps+1)), desc='Sampling ', total=self.n_timesteps):
        t = torch.full((batch_size,), i-1, dtype=torch.long, device=device)

        img = self.reverse_sample(img, t, i-1)

        if return_all_steps:
          imgs.append(img.cpu())

      if return_all_steps:
        return torch.stack(imgs).clamp(-1, 1)

      return img.clamp(-1, 1)
