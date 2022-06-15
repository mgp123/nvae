from torch import nn
import torch
from model.residual_cell_encoder import ResidualCellEncoder

from model.squeeze_excitation import SqueezeExcitation

class PreprocessCell(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super(PreprocessCell,self).__init__()
        self.model = ResidualCellEncoder(in_channels,out_channels,stride)

    def forward(self, x):
        return self.model(x)
 
class Preprocess(nn.Module):
    def __init__(self, in_channels, num_blocks, num_cells_per_block, channel_multiplier):
        super(Preprocess,self).__init__()
        model = [nn.Identity()]
        out_channels = in_channels*channel_multiplier
        for _ in range(num_blocks):
            for _ in range(num_cells_per_block-1):
                model.append(
                    PreprocessCell(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1
                        ))
            if num_cells_per_block != 0:
                model.append(
                    PreprocessCell(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2
                        )
                    )
            in_channels = out_channels
            out_channels = out_channels*channel_multiplier

        self.model = nn.Sequential(*model) 

    def forward(self, x):
        return self.model(x)
        
