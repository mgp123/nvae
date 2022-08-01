from torch import nn
import torch

# the original implementation also performs weight normalization on the kernels 
# TODO check if implementation is correct
class ARConv2d(nn.Conv2d):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                padding=0,
                groups=1,
                mirror=False,
                zero_diag=False):
        super(ARConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups
        )

        if not (groups == 1 or groups == in_channels):
            raise ValueError


        kernels_shape = out_channels, in_channels//groups, kernel_size,kernel_size

        self.mask = torch.ones(kernels_shape)

        half_kernel = (kernel_size-1)//2
        self.mask[:,:,half_kernel:,:] = 0
        self.mask[:,:,half_kernel,half_kernel:] = 1
        self.mask[:,:,half_kernel,half_kernel] = 1

        if groups == 1:
            repeating_dim = 1 if out_channels >= in_channels else 0
            oposite_dim = (repeating_dim + 1 ) % 2
            ratios = kernels_shape[oposite_dim]// kernels_shape[repeating_dim]
            center_mask = torch.tril(torch.ones((kernels_shape[repeating_dim], kernels_shape[repeating_dim])))

            if zero_diag:
                center_mask -=  torch.diag(torch.diag(center_mask))

            center_mask =  center_mask.repeat_interleave(ratios,dim=oposite_dim)
            self.mask[:,:,half_kernel,half_kernel] = center_mask

        
        if mirror:
            self.mask = torch.flip(self.mask, dims=[2,3])

    def forward(self, z):
        # return 2*z
        m = self.mask.to(z.device)
        return self._conv_forward(z,m*self.weight)



class NormalFlowCell(nn.Module):
    def __init__(self, latent_channel, hidden_channel_multiplier=6, mirror=False):
        super(NormalFlowCell, self).__init__()
        hidden_channels = latent_channel * hidden_channel_multiplier
        self.model = nn.Sequential(
            ARConv2d( 
                in_channels=latent_channel,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                mirror=mirror,
                zero_diag=True
            )
            ,
            nn.ELU(inplace=True),
            ARConv2d( 
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=5,
                padding=2,
                groups=hidden_channels,
                mirror=mirror
            ),
            nn.ELU(inplace=True),
            nn.ELU(),  # is the second ELU really necessary?
            ARConv2d(
                in_channels=hidden_channels,
                out_channels=latent_channel,
                kernel_size=1,
                mirror=mirror
            ),
        )

    def forward(self, z):
        y = self.model(z)
        return z - y


class NormalFlowBlock(nn.Module):
    def __init__(self, latent_channel, n_flows, n_blocks=1, hidden_channel_multiplier=6):
        super(NormalFlowBlock, self).__init__()
        # in our immplemenation we dont really need to differentiate between n_blocks and n_flows
        # we do it anyway to be consistent we the source
        normal_cells = []
        for _ in range(n_flows * n_blocks):
            normal_cells.append(
                NormalFlowCell(latent_channel, hidden_channel_multiplier)
            )
            normal_cells.append(
                NormalFlowCell(latent_channel, hidden_channel_multiplier, mirror=True)
            )
        self.model = nn.Sequential(*normal_cells)

    def forward(self, z, m):
        return self.model(z)
