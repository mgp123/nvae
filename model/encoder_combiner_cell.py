from turtle import forward
from torch import nn
import torch


# the combiner requieres both inputs to have the same shape except for the ammount of channels 
# - does a 1x1 convolution on the second input changing the channels to input_channels_2
# - adds it to the first input
class EncoderCombinerCell(nn.Module):
    """
    Used for mixing the output of the encoder part and the decoder part for the latent scale
    """
    def __init__(self, in_channels_1, in_channels_2):
        super(EncoderCombinerCell, self).__init__()
        self.model = nn.Conv2d(
            in_channels=in_channels_2, 
            out_channels=in_channels_1,
            kernel_size=(1, 1),
            )

    def forward(self, x1, x2):
        y = self.model(x2)
        return x1 + y
