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
    
    def regularization_loss(self):
        return self.model.regularization_loss()

    def get_batchnorm_cells(self):
        return self.model.get_batchnorm_cells()

 
class Preprocess(nn.Module):
    def __init__(self, in_channels: int, num_blocks: int, num_cells_per_block: int, channel_multiplier: int):
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

    def regularization_loss(self):
        loss = 0
        for i, l in enumerate(self.model):
            if i != 0:
                loss += l.regularization_loss()
        return loss
        
    def get_batchnorm_cells(self):
        res = []
        for i, l in enumerate(self.model):
            if i != 0:            
                res += l.get_batchnorm_cells()
        return res