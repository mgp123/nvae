import torch
import torch.nn as nn
from model.skip_cell import SkipCell

from model.squeeze_excitation import SqueezeExcitation
from model.utils import regularization_conv2d


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
                padding=1,
                # bias=False # TODO bias should be false
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
        return skip + 0.1 * self.model(x)

    def get_batchnorm_cells(self):
        return [self.model[0], self.model[3]]

    def get_conv_cells(self):
        return [self.model[2], self.model[5]]

    def regularization_loss(self):
        loss = 0
        for b_layer in self.get_batchnorm_cells():
            loss += torch.max(torch.abs(b_layer.weight))

        for c_layer in self.get_conv_cells():
            loss += regularization_conv2d(c_layer)

        return loss
