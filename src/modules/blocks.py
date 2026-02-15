import torch
import torch.nn as nn
import torch.nn.functional as f

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t_emb=None):
      h = self.relu(self.bn1(self.conv1(x)))

      if t_emb is not None:
        h = h + t_emb[:, :, None, None]

      h = self.relu(self.bn2(self.conv2(h)))

      return h


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, t_emb):
        # reduce the spatial dimensions by half and then pass through double convolution
        x = self.maxpool(x)
        return self.conv(x, t_emb)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # currently only using bilinear interpolation for upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection, t_emb):
        x = self.upsample(x) # doubles the spatial dimensions
        x = torch.cat([skip_connection, x], dim=1) # concatenate with skip connections

        return self.conv(x, t_emb)