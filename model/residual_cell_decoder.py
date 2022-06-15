from numpy import pad
import torch
import torch.nn as nn
from model.skip_cell import SkipCell
from model.squeeze_excitation import SqueezeExcitation


class ResidualCellDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channel_multiplier=6, stride=1):
        super(ResidualCellDecoder, self).__init__()

        inside_channels = in_channels * hidden_channel_multiplier
        up  = nn.UpsamplingNearest2d(scale_factor=2) if stride==-1 else nn.Identity()
        self.model = nn.Sequential(
            up,
            nn.BatchNorm2d(in_channels,  eps=1e-5, momentum=0.05),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inside_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(inside_channels, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=inside_channels,
                out_channels=inside_channels,
                kernel_size=(5, 5),
                padding=2,
                groups=inside_channels,
                bias=False
            ),            
            nn.BatchNorm2d(inside_channels,  eps=1e-5, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=inside_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum=0.05),
            SqueezeExcitation(out_channels)

        )

        self.skip = SkipCell(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
            )

    def forward(self, x):
        skip = self.skip(x)
        return skip + 0.1 * self.model(x)