from torch import nn
from model.encoder_block import EncoderBlock

from model.splitter_concatenate import SplitterConcatenate


class EncoderTower(nn.Module):
    def __init__(self, in_channels, number_of_scales, initial_splits_per_scale, cells_per_split=2, channel_multiplier=2,
                 exponential_decay_splits=1,min_splits=1):
        super(EncoderTower, self).__init__()

        self.model = SplitterConcatenate()


        number_of_splits = initial_splits_per_scale
        i_channels = in_channels
        o_channels = in_channels * channel_multiplier


        for i in range(number_of_scales):
            self.model.add_module(
                "enc_block_" + str(i + 1),
                EncoderBlock(
                    in_channels=i_channels,
                    continuation_channels=o_channels,
                    cells_per_split=cells_per_split,
                    number_of_splits=number_of_splits,
                    downsample=i != number_of_scales - 1  # don't downsample last block
                )
            )

            i_channels = o_channels
            o_channels *= channel_multiplier
            number_of_splits = max(min_splits, number_of_splits // exponential_decay_splits)

    def forward(self, x):
        y = self.model(x)
        y.reverse()
        return y
