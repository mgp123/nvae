from torch import nn
from model.encoder_group import EncoderGroup

from model.splitter_concatenate import SplitterConcatenate


class EncoderTower(nn.Module):
    def __init__(self, in_channels: int, number_of_scales: int, initial_splits_per_scale: int, cells_per_split=2, channel_multiplier=2,
                 exponential_decay_splits=1, min_splits=1):
        super(EncoderTower, self).__init__()

        self.model = nn.ModuleList()

        number_of_splits = initial_splits_per_scale
        i_channels = in_channels
        o_channels = in_channels * channel_multiplier

        for i in range(number_of_scales):
            self.model.add_module(
                "enc_block_" + str(i + 1),
                EncoderGroup(
                    in_channels=i_channels,
                    continuation_channels=o_channels if i != 0 else i_channels,
                    cells_per_split=cells_per_split,
                    number_of_splits=number_of_splits,
                    downscale=i != 0  # don't downscale before first group
                )
            )
            if i != 0:
                i_channels = o_channels
                o_channels *= channel_multiplier

            number_of_splits = max(min_splits, number_of_splits // exponential_decay_splits)

    def forward(self, x,use_tensor_checkpoints=False):
        y = []
        last_output = x
        for module in self.model:
            l = module(last_output,use_tensor_checkpoints=use_tensor_checkpoints)
            last_output = l[-1]
            y += l        
        y.reverse()
        return y

    def get_batchnorm_cells(self):
        res = []
        for l in self.model:
            res += l.get_batchnorm_cells()
        return res

    def regularization_loss(self):
        loss = 0
        for l in self.model:
            loss += l.regularization_loss()
        return loss
