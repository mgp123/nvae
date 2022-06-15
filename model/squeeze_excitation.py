import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, bottleneck_channels_ratio=2):
        super(SqueezeExcitation, self).__init__()

        bottleneck_channels = input_channels // bottleneck_channels_ratio
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.model = nn.Sequential(
            nn.Linear(input_channels, bottleneck_channels),
            nn.ReLU(),
            nn.Linear(bottleneck_channels, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.pooling(x)
        s = s.squeeze(3).squeeze(2)
        s = self.model(s).unsqueeze(2).unsqueeze(3)


        return x*s
