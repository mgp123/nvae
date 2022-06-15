from torch import nn

from model.residual_cell_decoder import ResidualCellDecoder

# TODO fix flow of up blocks. I think the upsample cell should go at the beginning and not at the end

class ResidualChain(nn.Module):
    def __init__(self, in_channels, out_channels, number_of_cells, upsample=False):
        super(ResidualChain,self).__init__()
        self.model = nn.Sequential()

        for i in range(number_of_cells):
            # the last cell is the one that changes the channels when not upsampling
            # otherwise this is done by the upsample cell
            o_channels = out_channels if (not upsample and i == number_of_cells - 1) else in_channels
            self.model.add_module(
                "cell_decoder" + str(i + 1),
                ResidualCellDecoder(
                    in_channels=in_channels,
                    out_channels=o_channels
                ))
        if upsample:
            self.model.add_module(
                "cell_decoder_upsample",
                ResidualCellDecoder(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=-1
                ))

    def forward(self, x):
        return self.model(x)

class DecoderTower(nn.Module):
    def __init__(self, out_channels, number_of_scales, final_inputs_per_scale, cells_per_input=1,
                 channel_divider=2, exponential_growth_inputs=1,min_splits=1):
        super(DecoderTower,self).__init__()

        blocks = []
        inputs = final_inputs_per_scale

        i_channels = out_channels
        o_channels = out_channels

        self.n_inputs = 0

        # we build the tower in reverse
        for i in range(number_of_scales):
            for inputs_in_scale in range(inputs):
                # only the last residual chain should upsample, also the last i doesnt upsample at the end
                upsample = inputs_in_scale == 0 and i != 0
                self.n_inputs += 1

                blocks.append(
                    ResidualChain(
                        in_channels=i_channels,
                        out_channels=i_channels if not upsample else o_channels,
                        number_of_cells=cells_per_input,
                        upsample=upsample
                    )
                )

            o_channels = i_channels
            i_channels = o_channels * channel_divider
            inputs = max(min_splits,inputs // exponential_growth_inputs)

        blocks.reverse()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, level):
        return self.blocks[level](x)
