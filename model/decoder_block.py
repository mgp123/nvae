from torch import nn

from model.residual_cell_decoder import ResidualCellDecoder


class ResidualChain(nn.Module):
    def __init__(self, in_channels, number_of_cells):
        super(ResidualChain).__init__()
        self.model = nn.Sequential()

        channels = in_channels
        for i in range(number_of_cells):
            self.model.add_module(
                "cell_decoder"+str(i+1),
                ResidualCellDecoder(
                    in_channels=channels,
                    out_channels=channels
                ))

    def forward(self, x):
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, continuation_channels, cells_per_split, number_of_splits, downsample=True):
        """
        Block of the decoder
        cells_per_split: number of consecutive residuals cells before doing a split
        number_of_splits: number of splits not including the last splitting which is used for downsampling
        continuation_channels: the number of channels for the downsample output
        downsample: optionally you can remove the last splitting associated with the downsample
        """
        super(DecoderBlock, self).__init__()

        self.model = None

    def forward(self, x):
        return self.model(x)
