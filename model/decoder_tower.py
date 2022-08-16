from torch import nn
import torch
from model.residual_cell_decoder import ResidualCellDecoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, number_of_cells, upscale=False):
        super(DecoderBlock, self).__init__()
        self.model = nn.Sequential()

        if upscale:
            self.model.add_module(
                "cell_decoder_upsample",
                ResidualCellDecoder(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=-1
                ))

        for i in range(number_of_cells):
            # the last cell is the one that changes the channels when not upscaling
            # otherwise this is done by the upscale cell

            # Note that we are never going to actually change the channels unless upscaling but to be consistent we add
            # that possibility
            i_channels = out_channels if upscale else in_channels
            o_channels = out_channels if (not upscale and i == number_of_cells - 1) else i_channels
            self.model.add_module(
                "cell_decoder" + str(i + 1),
                ResidualCellDecoder(
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

    def get_batchnorm_cells(self):
        res = []
        for l in self.model:
            res += l.get_batchnorm_cells()
        return res


class DecoderTower(nn.Module):
    def __init__(self, out_channels: int, number_of_scales: int, final_inputs_per_scale: int, cells_per_input=1,
                 channel_divider=2, exponential_growth_inputs=1, min_splits=1):
        super(DecoderTower, self).__init__()

        blocks = []
        inputs = final_inputs_per_scale

        i_channels = out_channels
        o_channels = out_channels

        self.n_inputs = 0

        # we build the tower in reverse
        for i in range(number_of_scales):
            for inputs_in_scale in range(inputs):
                # only the last block from group should upscale, also the last i doesnt upscale at the end
                upscale = inputs_in_scale == 0 and i != 0
                self.n_inputs += 1

                if not (i == 0 and inputs_in_scale == 0):
                    blocks.append(
                        DecoderBlock(
                            in_channels=i_channels,
                            out_channels=i_channels if not upscale else o_channels,
                            number_of_cells=cells_per_input,
                            upscale=upscale
                        )
                    )
                else:
                    # the combination of the encoder and decoder towers end with a mixing so we set 
                    # the last block as Identity
                    blocks.append(nn.Identity())

            o_channels = i_channels
            i_channels = o_channels * channel_divider

            inputs = max(min_splits, inputs // exponential_growth_inputs)

        blocks.reverse()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, level, use_tensor_checkpoints=False):
        if use_tensor_checkpoints:
            # this messes up the batchnorm
            x =  torch.utils.checkpoint.checkpoint(self.blocks[level], x)
            return x
        else:
            return self.blocks[level](x)

    def get_batchnorm_cells(self):
        res = []
        for i, l in enumerate(self.blocks):
            if i != len(self.blocks) - 1:
                res += l.get_batchnorm_cells()
        return res

    def regularization_loss(self):
        loss = 0
        for i, l in enumerate(self.blocks):
            if i != len(self.blocks) - 1:
                loss += l.regularization_loss()
        return loss