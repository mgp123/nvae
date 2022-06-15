import torch.nn as nn
from model.skip_cell import SkipCell

from model.squeeze_excitation import SqueezeExcitation


class ResidualCellEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualCellEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1
            ),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            ),
            SqueezeExcitation(out_channels)
        )

        self.skip = SkipCell(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
            )

    def forward(self, x):
        skip = self.skip(x)
        return skip + 0.1*self.model(x)
