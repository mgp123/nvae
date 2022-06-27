from torch import nn

from model.residual_cell_encoder import ResidualCellEncoder
from model.splitter import Splitter


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, continuation_channels, number_of_cells, downscale=False):
        super(EncoderBlock, self).__init__()
        self.model = nn.Sequential()

        if downscale:
            self.model.add_module(
                "down",
                ResidualCellEncoder(
                    in_channels=in_channels,
                    out_channels=continuation_channels,
                    stride=2
                )
            )
        for i in range(number_of_cells):
            # the last cell is the one that changes the channels when not downscale
            # otherwise this is done by the upscale cell

            # Note that we are never going to actually change the channels unless downscaling but to be consistent we
            # add that possibility
            i_channels = continuation_channels if downscale else in_channels
            o_channels = continuation_channels if (not downscale and i == number_of_cells - 1) else i_channels

            self.model.add_module(
                "cell_encoder" + str(i + 1),
                ResidualCellEncoder(
                    in_channels=i_channels,
                    out_channels=o_channels
                ))

    def forward(self, x):
        return self.model(x)

    def regularization_loss(self):
        loss = 0
        for l in self.model:
            loss += l.regularization_loss()
        return loss

class EncoderGroup(nn.Module):
    def __init__(self, in_channels, continuation_channels, cells_per_split, number_of_splits, downscale=True):
        """
        Group of the encoder
        cells_per_split: number of consecutive residuals cells before doing a split
        number_of_splits: number of splits in this group
        continuation_channels: the number of channels for the downscale output
        downscale: optionally you can avoid downscaling with the first block
        """
        super(EncoderGroup, self).__init__()

        self.model = Splitter()
        out_channels = continuation_channels

        for i in range(number_of_splits):

            self.model.add_module(
                "split_" + str(i + 1),
                EncoderBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    continuation_channels=out_channels,
                    number_of_cells=cells_per_split,
                    downscale=downscale and i == 0,
                )
            )

    def forward(self, x):
        return self.model(x)

    def regularization_loss(self):
        loss = 0
        for l in self.model:
            loss += l.regularization_loss()
        return loss