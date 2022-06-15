from torch import nn

from model.residual_cell_encoder import ResidualCellEncoder
from model.splitter import Splitter


class ResidualChain(nn.Module):
    def __init__(self, in_channels, number_of_cells):
        super(ResidualChain,self).__init__()
        self.model = nn.Sequential()

        for i in range(number_of_cells):
            self.model.add_module(
                "cell_encoder"+str(i+1),
                ResidualCellEncoder(
                    in_channels=in_channels,
                    out_channels=in_channels
                ))

    def forward(self, x):
        return self.model(x)

# TODO fix flow of up blocks. I think the down cell should go at the beginning and not at the end
# also down should be sequential with ResidualChain and not a separete part of splitter
# similarly splitter should not merge last part 
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, continuation_channels, cells_per_split, number_of_splits, downsample=True):
        """
        Block of the encoder
        cells_per_split: number of consecutive residuals cells before doing a split
        number_of_splits: number of splits not including the last splitting which is used for downsampling
        continuation_channels: the number of channels for the downsample output
        downsample: optionally you can remove the last splitting associated with the downsample
        """
        super(EncoderBlock, self).__init__()

        self.model = Splitter()

        for i in range(number_of_splits):
            self.model.add_module(
                "split_" + str(i+1),
                ResidualChain(
                    in_channels=in_channels,
                    number_of_cells=cells_per_split
                )
            )
        if downsample:
            self.model.add_module(
                "down",
                ResidualCellEncoder(
                    in_channels=in_channels,
                    out_channels=continuation_channels,
                    stride=2
                )
            )


    def forward(self, x):
        return self.model(x)

