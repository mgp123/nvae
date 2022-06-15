from torch import nn

class NormalFlowCell(nn.Module):
    def __init__(self, latent_channel, hidden_channel_multiplier=6):
        super(NormalFlowCell, self).__init__()
        hidden_channels = latent_channel* hidden_channel_multiplier
        self.model = nn.Sequential(
            nn.Conv2d( # TODO change for masked convolution 
                in_channels= latent_channel,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d( # TODO change for masked convolution 
                in_channels= hidden_channels,
                out_channels=hidden_channels,
                kernel_size=5,
                padding=2,
                groups=hidden_channels),
            nn.ELU(inplace=True),
            nn.ELU(), # is the second ELU really necessary?
            nn.Conv2d( # TODO change for masked convolution 
                in_channels= hidden_channels,
                out_channels=latent_channel,
                kernel_size=1,
                ),
        )

    def forward(self, z):
        y = self.model(z)
        return z-y

class NormalFlowBlock(nn.Module):
    def __init__(self, latent_channel, n_flows, n_blocks=2, hidden_channel_multiplier=6):
        super(NormalFlowBlock,self).__init__()
        # in our immplemenation we dont really need to differentiate between n_blocks and n_flows
        # we do it anyway to be consistent we the source
        normal_cells = []
        for _ in range(n_flows* n_blocks):
            normal_cells.append(
                NormalFlowCell(latent_channel, hidden_channel_multiplier)
                )
        self.model = nn.Sequential(*normal_cells)
    

    def forward(self, z ,m):
        return self.model(z)


