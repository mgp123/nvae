from tkinter import X
from typing import List, Union
from torch import nn
import torch
from torch.cuda.amp import autocast

from model.decoder_tower import DecoderTower
from model.encoder_tower import EncoderTower
from model.mixer import Mixer
from model.postprocess import Postprocess
from model.preprocess import Preprocess
from model.distributions import Normal, DiscMixLogistic, Distribution


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
                 sampling_method="gaussian",
                 n_mix=3,
                 fixed_flows=False):
        super(Autoencoder, self).__init__()

        self.cached_batch = None
        self.latent_size = latent_size
        self.sampling_method = sampling_method
        self.initial_splits_per_scale = initial_splits_per_scale
        self.number_of_scales = number_of_scales
        self.exponential_scaling = exponential_scaling
        self.min_splits = min_splits
        self.input_dimension = input_dimension
        self.num_blocks_prepost = num_blocks_prepost
        self.use_tensor_checkpoints = False

        channels_towers_inside = channel_towers * (channel_multiplier ** num_blocks_prepost)

        self.initial_transform = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels=3,
                out_channels=channel_towers,
                kernel_size=3,
                padding=1
            )
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
            min_splits=min_splits,
            fixed_flows=fixed_flows
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

        d_parameters = \
            3 * \
            Distribution.get_class(sampling_method).params_per_subpixel(n_mix)

        self.to_distribution_conv = nn.Sequential(
            nn.ELU(),
            nn.utils.weight_norm(
                nn.Conv2d(
                    channel_towers,
                    d_parameters,
                    3, padding=1, bias=True
                )
            )
        )

    def set_use_tensor_checkpoints(self, value):
        self.use_tensor_checkpoints = value

    def forward(self, x):
        enc_parts = \
            self.encoder_tower(
                self.preprocess(
                    self.initial_transform(x)
                ),
                use_tensor_checkpoints=self.use_tensor_checkpoints
            )
        kl_losses = []
        residual_dec = self.decoder_constant.expand((enc_parts[0].size(0), -1, -1, -1))
        for i, enc_part_i in enumerate(enc_parts):
            # note that the mixing for i == 0 behaves different
            residual_dec, kl_loss = self.mixer(enc_part_i, residual_dec, i)
            kl_losses.append(kl_loss)
            residual_dec = self.decoder_tower(residual_dec, i, use_tensor_checkpoints=self.use_tensor_checkpoints)

        distribution_params = self.to_distribution_conv(
            self.postprocess(residual_dec)
        )

        x_distribution = Distribution.construct_from_params(
            self.sampling_method,
            distribution_params
        )

        return x_distribution, kl_losses

    def sample(self, n: int, t: Union[List[float], float] = 1.0, final_distribution_sampling="mean"):
        ts = t

        if isinstance(t, float) or isinstance(t, int):
            ts = [t] * self.decoder_tower.n_inputs

        residual_dec = self.decoder_constant.expand((n, -1, -1, -1))
        for i in range(self.decoder_tower.n_inputs):
            # note that the mixing for i == 0 behaves different
            residual_dec = self.mixer.decoder_only_mix(residual_dec, i, t=ts[i])
            residual_dec = self.decoder_tower(residual_dec, i)

        distribution_params = self.to_distribution_conv(
            self.postprocess(residual_dec)
        )

        x_distribution = Distribution.construct_from_params(
            self.sampling_method,
            distribution_params
        )

        x = x_distribution.get(final_distribution_sampling)

        x = torch.clamp(x, 0, 1.)

        return x

    def generate_from_latents(self, zs, final_distribution_sampling="mean"):
        batch_size = zs[0].shape[0]
        residual_dec = self.decoder_constant.expand((batch_size, -1, -1, -1))
        for i, z in enumerate(zs):
            residual_dec = self.mixer.from_normalized_latent_mix(residual_dec, i, z)
            residual_dec = self.decoder_tower(residual_dec, i)

        distribution_params = self.to_distribution_conv(
            self.postprocess(residual_dec)
        )

        x_distribution = Distribution.construct_from_params(
            self.sampling_method,
            distribution_params
        )
        x = x_distribution.get(final_distribution_sampling)
        x = torch.clamp(x, 0, 1.)

        return x

    

    def encode(self, x) -> List[torch.Tensor]:
        """
        Note that this is a non-deterministic encoding.
        That is, each run may give you a different set of tensors
        It outputs a list of tensors z, each of these the one associated with the mixing with the decoder
        """
        enc_parts = \
            self.encoder_tower(
                self.preprocess(
                    self.initial_transform(x)
                ),
            )

        zs = []
        residual_dec = self.decoder_constant.expand((enc_parts[0].size(0), -1, -1, -1))

        for i, enc_part_i in enumerate(enc_parts):
            # note that the mixing for i == 0 behaves different
            residual_dec, z = self.mixer.mix_and_get_z(enc_part_i, residual_dec, i)
            zs.append(z)
            residual_dec = self.decoder_tower(residual_dec, i)
        return zs

    def decode(self, zs, final_distribution_sampling="mean") -> torch.Tensor:
        n = zs[0].shape[0]
        residual_dec = self.decoder_constant.expand((n, -1, -1, -1))
        for i, z in enumerate(zs):
            # note that the mixing for i == 0 behaves different
            residual_dec = self.mixer.mix_with_z(residual_dec, z, i)
            residual_dec = self.decoder_tower(residual_dec, i)

        distribution_params = self.to_distribution_conv(
            self.postprocess(residual_dec)
        )

        x_distribution = Distribution.construct_from_params(
            self.sampling_method,
            distribution_params
        )

        x = x_distribution.get(final_distribution_sampling)

        x = torch.clamp(x, 0, 1.)
        return x

    def get_batchnorm_cells(self):
        if self.cached_batch is None:
            self.cached_batch = self.encoder_tower.get_batchnorm_cells() + \
                                self.decoder_tower.get_batchnorm_cells() + \
                                self.preprocess.get_batchnorm_cells() + \
                                self.postprocess.get_batchnorm_cells() + \
                                self.mixer.get_batchnorm_cells()
        return self.cached_batch

    def regularization_loss(self):
        # cached_batch = self.get_batchnorm_cells()
        # k = list(map(lambda x: torch.max(torch.abs(x.weight)), cached_batch))
        # k = torch.stack(k)
        # return torch.sum(k)

        loss = self.encoder_tower.regularization_loss() + \
               self.decoder_tower.regularization_loss() + \
               self.preprocess.regularization_loss() + \
               self.postprocess.regularization_loss() + \
               self.mixer.regularization_loss()
        return loss
