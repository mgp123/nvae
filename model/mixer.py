from typing import Tuple
from torch import nn
import torch
from model.normal_flow_block import NormalFlowBlock

from model.utils import device, regularization_conv2d
from model.distributions import Normal


# we are always going to have in_channels1==in_channels2
# but to be consistent with the original implementation we
# add the flexibility of having different in_channels1 and in_channels2
class MixerCellEncoder(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(MixerCellEncoder, self).__init__()
        self.model = nn.utils.weight_norm(
            nn.Conv2d(in_channels=in_channels2, out_channels=in_channels1, kernel_size=1)
        )

    def forward(self, x1, x2):
        return x1 + self.model(x2)


class MixerCellDecoder(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(MixerCellDecoder, self).__init__()
        self.model = nn.utils.weight_norm(
            nn.Conv2d(in_channels=in_channels1 + in_channels2, out_channels=in_channels1, kernel_size=1)
        )

    def forward(self, x1, x2):
        y = torch.cat([x1, x2], dim=1)
        return self.model(y)


class DummyMixer(nn.Module):
    def __init__(self, in_channels):
        super(DummyMixer, self).__init__()

        # pytorch weight normalization is prone to generating nans when the norm becomes small!
        #  there's no safety coeffcient added

        self.model = nn.Sequential(
            nn.ELU(),
            nn.utils.weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)),
            nn.ELU()
        )

    def forward(self, x1, x2):
        y = self.model(x1)
        return y


class Mixer(nn.Module):
    def __init__(self,
                 channels_towers,
                 number_of_scales,
                 initial_splits_per_scale,
                 latent_size,
                 n_flows,
                 channel_multiplier=2,
                 exponential_scaling=1,
                 min_splits=1,
                 fixed_flows=True):

        super(Mixer, self).__init__()

        self.latent_size = latent_size
        self.fixed_flows = fixed_flows  # or n_flows == 0
        encoder_sampler = []
        decoder_sampler = []
        encoder_mixer = []
        decoder_mixer = []
        normal_flow_block = []
        number_of_splits = initial_splits_per_scale
        i_channels = channels_towers
        o_channels = channels_towers * channel_multiplier

        for i in range(number_of_scales):
            for k in range(number_of_splits):
                # the first mixer is special
                if not (i == number_of_scales - 1 and k == number_of_splits - 1):
                    encoder_mixer.append(
                        MixerCellEncoder(i_channels, i_channels)
                    )

                    decoder_sampler.append(
                        nn.Sequential(
                            nn.ELU(),
                            nn.utils.weight_norm(
                                nn.Conv2d(
                                    in_channels=i_channels,
                                    out_channels=latent_size * 2,
                                    kernel_size=1,
                                )
                            ),
                        )
                    )
                else:
                    encoder_mixer.append(
                        DummyMixer(i_channels)
                    )

                encoder_sampler.append(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=i_channels,
                            out_channels=latent_size * 2,
                            kernel_size=3,
                            padding=1
                        )
                    )
                )

                decoder_mixer.append(MixerCellDecoder(i_channels, latent_size))
                normal_flow_block.append(
                    NormalFlowBlock(
                        latent_channel=latent_size,
                        n_flows=n_flows
                    ))

            i_channels = o_channels
            o_channels *= channel_multiplier
            number_of_splits = max(min_splits, number_of_splits // exponential_scaling)

        decoder_sampler.append(None)  # not used. Added to avoid substracting from index

        encoder_mixer.reverse()
        encoder_sampler.reverse()
        decoder_mixer.reverse()
        decoder_sampler.reverse()
        normal_flow_block.reverse()

        self.encoder_sampler = nn.ModuleList(encoder_sampler)
        self.encoder_mixer = nn.ModuleList(encoder_mixer)
        self.decoder_sampler = nn.ModuleList(decoder_sampler)
        self.decoder_mixer = nn.ModuleList(decoder_mixer)
        self.normal_flow_block = nn.ModuleList(normal_flow_block)

    def forward(self, enc_part, dec_part, i):
        m = self.encoder_mixer[i](enc_part, dec_part)
        latent = self.encoder_sampler[i](m)

        # the first latent variable of the encoder 
        # is going to try to imitate the standard normal distribution
        latent_dec = torch.zeros_like(latent)

        if i != 0:
            latent_dec = self.decoder_sampler[i](dec_part)
            latent = latent + latent_dec

        distribution_enc = Normal(latent)
        distribution_dec = Normal(latent_dec)

        z = distribution_enc.sample()

        kl_loss = None
        if self.fixed_flows:
            kl_loss = torch.sum(distribution_enc.kl(distribution_dec), dim=[1, 2, 3])
            z = self.normal_flow_block[i](z, m)
        else:
            log_enc = torch.sum(distribution_enc.log_p(z), dim=[1, 2, 3])
            z = self.normal_flow_block[i](z, m)
            log_dec = torch.sum(distribution_dec.log_p(z), dim=[1, 2, 3])
            kl_loss = log_enc - log_dec

        y = self.decoder_mixer[i](dec_part, z)

        return y, kl_loss

    def decoder_only_mix(self, dec_part, i, t=1):
        latent = None
        if i != 0:
            latent = self.decoder_sampler[i](dec_part)
        else:
            batch_size, _, h, w = dec_part.size()
            latent = torch.zeros((batch_size, self.latent_size * 2, h, w)).to(device)

        distribution = Normal(latent, t=t)
        z = distribution.sample()

        if self.fixed_flows:
            z = self.normal_flow_block[i](z, None)

        y = self.decoder_mixer[i](dec_part, z)
        return y

    def from_normalized_latent_mix(self, dec_part, i, z):
        latent = None
        if i != 0:
            latent = self.decoder_sampler[i](dec_part)
        else:
            batch_size, _, h, w = dec_part.size()
            latent = torch.zeros((batch_size, self.latent_size * 2, h, w)).to(device)

        distribution = Normal(latent)
        z = distribution.normal_sample_transform(z)

        if self.fixed_flows:
            z = self.normal_flow_block[i](z, None)

        y = self.decoder_mixer[i](dec_part, z)
        return y

    def mix_and_get_z(self, enc_part, dec_part, i) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.encoder_mixer[i](enc_part, dec_part)
        latent = self.encoder_sampler[i](m)

        # the first latent variable of the encoder 
        # is going to try to imitate the standard normal distribution
        latent_dec = torch.zeros_like(latent)

        if i != 0:
            latent_dec = self.decoder_sampler[i](dec_part)
            latent = latent + latent_dec

        distribution_enc = Normal(latent)

        z = distribution_enc.sample()

        if self.fixed_flows:
            z = self.normal_flow_block[i](z, m)
        y = self.decoder_mixer[i](dec_part, z)

        return y, z

    def mix_with_z(self, dec_part, z, i) -> torch.Tensor:
        y = self.decoder_mixer[i](dec_part, z)
        return y

    def get_batchnorm_cells(self):
        res = []
        return res

    def regularization_loss(self):
        loss = 0
        for m in self.encoder_sampler:
            loss += regularization_conv2d(m)
        for m in self.decoder_sampler:
            if m is not None:
                loss += regularization_conv2d(m[1])

        return loss
