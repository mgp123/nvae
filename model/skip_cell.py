from torch import nn
import torch
import torch.nn.functional as F


# skip cell for stride = 2
class SkipCellDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        if out_channels % 4 != 0:
            raise ValueError("out_channels in SkipCellDown must be a multiple of four")

        super(SkipCellDown, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels // 4, 1, stride=2)
        self.conv_2 = nn.Conv2d(in_channels, out_channels // 4, 1, stride=2)
        self.conv_3 = nn.Conv2d(in_channels, out_channels // 4, 1, stride=2)
        self.conv_4 = nn.Conv2d(in_channels, out_channels // 4, 1, stride=2)
        self.silu = nn.SiLU()

    def forward(self, x):
        y = self.silu(x)

        # NVAE slice the first row, column or both in sucesive convolutions
        # what's the point of that?

        conv1 = self.conv_1(y)
        conv2 = self.conv_2(y)
        conv3 = self.conv_3(y)
        conv4 = self.conv_4(y)

        return torch.cat([conv1, conv2, conv3, conv4], dim=1)


class SkipCellUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipCellUp, self).__init__()

        self.model = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        y = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.model(y)


class SkipCell(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkipCell, self).__init__()
        self.model = nn.Sequential()
        if stride == 1 and in_channels == out_channels:
            self.model.add_module("same shape", nn.Identity())
        elif stride == 2:
            self.model.add_module("skip down", SkipCellDown(in_channels, out_channels))
            # raise NotImplementedError
        elif stride == -1:
            self.model.add_module("skip up", SkipCellUp(in_channels, out_channels))
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)
