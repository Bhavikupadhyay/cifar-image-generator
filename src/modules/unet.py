import torch
import math
import torch.nn as nn
from .blocks import DoubleConv, Downsample, Upsample

class TimeEmbeddings(nn.Module):
    def __init__(self, embedding_dim=256, max_time=1000):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_time = max_time

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('inv_freq', 1 / emb)

    def forward(self, t):
        emb = t[:, None] * self.inv_freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class UNet(nn.Module):
  def __init__(self, n_channels=3, n_classes=3, time_emb_dim=256, base_channels=64):
    super().__init__()

    self.time_mlp = nn.Sequential(
        nn.Linear(time_emb_dim, time_emb_dim),
        nn.GELU(),
        nn.Linear(time_emb_dim, time_emb_dim)
    )

    # Rename for ease of use
    c = base_channels

    # Encoders
    self.inc = DoubleConv(n_channels, c)
    self.down1 = Downsample(c, c * 2)
    self.down2 = Downsample(c * 2, c * 4)
    self.down3 = Downsample(c * 4, c * 8)

    # Bottleneck
    self.bot = DoubleConv(c * 8, c * 16)

    # Decoders
    self.up1 = Upsample((c * 16) + (c * 4), (c * 4))
    self.up2 = Upsample((c * 4) + (c * 2), (c * 2))
    self.up3 = Upsample(c * 2 + c, c)

    # Output
    self.outc = nn.Conv2d(c, n_classes, kernel_size=1)

    self.time_projections = nn.ModuleList([
        nn.Linear(time_emb_dim, c),    # inc
        nn.Linear(time_emb_dim, c * 2),   # down1
        nn.Linear(time_emb_dim, c * 4),   # down2
        nn.Linear(time_emb_dim, c * 8),   # down3
        nn.Linear(time_emb_dim, c * 16),  # bottleneck
        nn.Linear(time_emb_dim, c * 4),   # up1
        nn.Linear(time_emb_dim, c * 2),   # up2
        nn.Linear(time_emb_dim, c),    # up3
    ])

  def forward(self, x, t):
    t = self.time_mlp(t)

    def get_time(layer_idx, tensor_t):
      return self.time_projections[layer_idx](tensor_t)

    # Downsample block
    x1 = self.inc(x, get_time(0, t))
    x2 = self.down1(x1, get_time(1, t))
    x3 = self.down2(x2, get_time(2, t))
    x4 = self.down3(x3, get_time(3, t))

    # Bottleneck
    x = self.bot(x4, get_time(4, t))

    # Upsample block
    x = self.up1(x, x3, get_time(5, t))
    x = self.up2(x, x2, get_time(6, t))
    x = self.up3(x, x1, get_time(7, t))

    # Final Output
    return self.outc(x)