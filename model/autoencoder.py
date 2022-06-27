from torch import nn
import torch
from torch.cuda.amp import autocast

from model.decoder_tower import DecoderTower
from model.encoder_tower import EncoderTower
from model.mixer import Mixer
from model.postprocess import Postprocess
from model.preprocess import Preprocess
from model.utils import DiscMixLogistic, Normal


class Autoencoder(nn.Module):
    def __init__(self,
                 channel_towers,
                 number_of_scales,
                 initial_splits_per_scale,
                 latent_size,
                 input_dimension,
                 num_flows,
                 num_blocks_prepost=1,
                 num_cells_per_block_prepost=2,
                 cells_per_split_enc=2,
                 cells_per_input_dec=1,
                 channel_multiplier=2,
                 exponential_scaling=1,
                 min_splits=1,
                 sampling_method="gaussian"):
        super(Autoencoder, self).__init__()

        self.latent_size = latent_size
        self.sampling_method = sampling_method
        self.initial_splits_per_scale = initial_splits_per_scale
        self.number_of_scales = number_of_scales
        self.exponential_scaling = exponential_scaling
        self.min_splits = min_splits

        channels_towers_inside = channel_towers * (channel_multiplier ** (num_blocks_prepost))

        self.initial_transform = nn.Conv2d(
            in_channels=3,
            out_channels=channel_towers,
            kernel_size=3,
            padding=1
        )

        self.preprocess = Preprocess(
            channel_towers,
            num_blocks_prepost,
            num_cells_per_block_prepost,
            channel_multiplier)

        self.encoder_tower = \
            EncoderTower(
                in_channels=channels_towers_inside,
                number_of_scales=number_of_scales,
                initial_splits_per_scale=initial_splits_per_scale,
                cells_per_split=cells_per_split_enc,
                channel_multiplier=channel_multiplier,
                exponential_decay_splits=exponential_scaling,
                min_splits=min_splits
            )

        self.mixer = Mixer(
            channels_towers_inside,
            number_of_scales,
            initial_splits_per_scale,
            latent_size,
            num_flows,
            exponential_scaling=exponential_scaling,
            min_splits=min_splits
        )

        self.f_channels = channels_towers_inside * (channel_multiplier ** (number_of_scales - 1))
        self.f_dimension = input_dimension // (2 ** (number_of_scales + num_blocks_prepost - 1))
        self.decoder_constant = nn.Parameter(
            torch.randn((1, self.f_channels, self.f_dimension, self.f_dimension)))

        self.decoder_tower = \
            DecoderTower(
                out_channels=channels_towers_inside,
                number_of_scales=number_of_scales,
                final_inputs_per_scale=initial_splits_per_scale,
                cells_per_input=cells_per_input_dec,
                channel_divider=channel_multiplier,
                exponential_growth_inputs=exponential_scaling,
                min_splits=min_splits
            )

        self.postprocess = Postprocess(
            channels_towers_inside,
            num_blocks_prepost,
            num_cells_per_block_prepost,
            channel_multiplier
        )

        d_parameters = None
        if sampling_method == "mixture":
            d_parameters = 9*2
        elif sampling_method == "gaussian":
            d_parameters = 6  # 3 for channel mean, 3 for log_std
        else:
            raise ValueError

        self.to_distribution_conv = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(
                channel_towers,
                d_parameters,
                3, padding=1, bias=True)
        )

    def forward(self, x):
        enc_parts = \
            self.encoder_tower(
                self.preprocess(
                    self.initial_transform(x)
                )
            )
        kl_losses = []
        # TODO check if expanding makes each batch element separe and works with the learning of the constant 
        residual_dec = self.decoder_constant.expand((enc_parts[0].size(0), -1, -1, -1))
        for i, enc_part_i in enumerate(enc_parts):
            # note that the mixing for i == 0 behaves different
            residual_dec, kl_loss = self.mixer(enc_part_i, residual_dec, i)
            kl_losses.append(kl_loss)
            residual_dec = self.decoder_tower(residual_dec, i)

        x_distribution = self.to_distribution_conv(
            self.postprocess(residual_dec)
        )

        if self.sampling_method == "mixture":
            x_distribution = DiscMixLogistic(x_distribution)
            return x_distribution, kl_losses

        elif self.sampling_method == "gaussian":
            # we can see the per batch Normal distributions as a big whole batch Normal distribution
            mu, log_sig = torch.chunk(x_distribution, 2, 1)
            x_distribution = Normal(mu, log_sig)

            return x_distribution, kl_losses

    def sample(self, n, t=1):
        residual_dec = self.decoder_constant.expand((n, -1, -1, -1))
        for i in range(self.decoder_tower.n_inputs):
            # note that the mixing for i == 0 behaves different
            residual_dec = self.mixer.decoder_only_mix(residual_dec, i, t=t)
            residual_dec = self.decoder_tower(residual_dec, i)

        x_distribution = self.to_distribution_conv(
            self.postprocess(residual_dec)
        )

        x = None 

        if self.sampling_method == "mixture":
            x_distribution = DiscMixLogistic(x_distribution)
            x = x_distribution.sample()
        elif self.sampling_method == "gaussian":
            # we can see the per batch Normal distributions as a big whole batch Normal distribution
            mu, log_sig = torch.chunk(x_distribution, 2, 1)
            x_distribution = Normal(mu, log_sig)
            # we may stick with mu if we have a gaussian disttribution
            # note that we still generate x_distribution due to the soft_clamp
            x = x_distribution.mu

        x = torch.clamp(x, 0, 1.)

        # TODO should I remove this?
        # x = x / 2. + 0.5
        return x

    def regularization_loss(self):
        loss = self.encoder_tower.regularization_loss() + \
               self.decoder_tower.regularization_loss() + \
               self.preprocess.regularization_loss() + \
               self.postprocess.regularization_loss() + \
               self.mixer.regularization_loss()
        return loss
