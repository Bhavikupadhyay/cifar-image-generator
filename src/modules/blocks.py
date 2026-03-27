import torch
import torch.nn as nn

GN_GROUPS = 32  # channels must be divisible by this; holds for all widths with base_channels >= 64


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_group_norm=True):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self._use_gn = use_group_norm

        if use_group_norm:
            self.gn1 = nn.GroupNorm(GN_GROUPS, mid_channels)
            self.gn2 = nn.GroupNorm(GN_GROUPS, out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, t_emb=None):
        n1, n2 = (self.gn1, self.gn2) if self._use_gn else (self.bn1, self.bn2)
        h = self.relu(n1(self.conv1(x)))
        if t_emb is not None:
            h = h + t_emb[:, :, None, None]
        return self.relu(n2(self.conv2(h)))


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_group_norm=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, use_group_norm=use_group_norm)

    def forward(self, x, t_emb):
        return self.conv(self.maxpool(x), t_emb)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_group_norm=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, use_group_norm=use_group_norm)

    def forward(self, x, skip_connection, t_emb):
        x = self.upsample(x)
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x, t_emb)
