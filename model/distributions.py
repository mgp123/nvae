from __future__ import annotations
import math
from typing import Type
import torch
import torch.distributions
from model.utils import device


# taken directly from source

def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)


logistic_distribution = torch.distributions.TransformedDistribution(
    torch.distributions.uniform.Uniform(1e-5, 1. - 1e-5),
    [torch.distributions.SigmoidTransform().inv]
)


def logits_to_log_probs(logits):
    m_logits, _ = torch.max(logits, dim=2, keepdim=True)
    log_probs = torch.log(
        torch.sum(
            torch.exp(logits - m_logits),
            dim=2, keepdim=True))
    log_probs = logits - m_logits - log_probs
    return log_probs


# pytorch should add a torch.pi !
log2pi = math.log(2 * math.pi)


class Distribution:

    def __init__(self) -> None:
        pass

    def mean(self):
        pass

    def most_likely_value(self):
        pass

    def sample(self):
        pass

    @classmethod
    def params_per_subpixel(cls, *args, **kwargs) -> int:
        pass

    def get(self, statistic) -> torch.Tensor:
        if statistic == "mean":
            return self.mean()
        elif statistic == "most_likely_value":
            return self.most_likely_value()
        elif statistic == "sample":
            return self.sample()
        else:
            raise ValueError

    @classmethod
    def get_class(cls, class_name) -> Type[Distribution]:
        if class_name == "gaussian":
            return Normal
        elif class_name == "logistic_mixture":
            return DiscMixLogistic
        elif class_name == "logistic":
            return DiscLogistic
        elif class_name == "gaussian_mixture":
            return NormalMix
        else:
            raise ValueError

    @classmethod
    def construct_from_params(cls, class_name, params) -> Distribution:
        return Distribution.get_class(class_name)(params)


# slightly modified version of Normal class in source

class Normal(Distribution):
    """
    Diagonal batch-wise gaussians
    """

    def __init__(self, params, t=1):
        super().__init__()
        mu, log_sig = torch.chunk(params, 2, 1)
        self.mu = soft_clamp5(mu)
        # soft_clamp5 log_sig?
        log_sig = soft_clamp5(log_sig)

        self.sig = torch.exp(log_sig) * t + 1e-2

    # takes sample z following N(0,I) and transforms it into a sample from this normal
    def normal_sample_transform(self, z):
        return z * self.sig + self.mu

    def sample(self):
        super().__init__()

        return self.normal_sample_transform(torch.randn_like(self.mu))

    # log probability of observing z_i, each feature of z is done independently.
    # We ignore constants here
    def log_p(self, z):
        normalized_samples = (z - self.mu) / self.sig
        log_p = - 0.5 * normalized_samples * normalized_samples - torch.log(self.sig)

        # we dont really need to add the constant if we are going to use this as the loss
        # but we add it just to be consisten with Normal mixture
        log_p -= 0.5 * log2pi

        return log_p

    def kl(self, normal):
        log_det1 = torch.log(self.sig)
        log_det2 = torch.log(normal.sig)

        inv_sigma2 = 1.0 / normal.sig

        delta_mu = normal.mu - self.mu

        # trace_loss = torch.einsum('bcwh,bcwh->b', self.sig, inv_sigma2)
        # mu_loss = torch.einsum('bcwh,bcwh->b', delta_mu, delta_mu * inv_sigma2)

        # trace_loss = torch.einsum('b..., b... -> b', self.sig, inv_sigma2)
        # mu_loss = torch.einsum('b..., b... -> b', delta_mu, delta_mu * inv_sigma2)

        # trace_loss = torch.sum(self.sig * inv_sigma2, dim=dims)
        # mu_loss = torch.sum(delta_mu * inv_sigma2 * delta_mu, dim=dims)

        det_loss = log_det2 - log_det1
        kl_loss = (torch.square(delta_mu) + torch.square(self.sig))
        kl_loss = kl_loss * torch.square(inv_sigma2) - 1
        kl_loss = det_loss + 0.5 * kl_loss

        return kl_loss

    def mean(self):
        super().__init__()
        return self.mu

    def most_likely_value(self):
        super().__init__()
        return self.mu

    @classmethod
    def params_per_subpixel(cls, *args, **kwargs):
        return 2


class NormalMix(Distribution):
    def __init__(self, params):
        batch_size, c, height, width = params.size()
        n_mix = c // 9
        l = soft_clamp5(params.view(batch_size, 3, 3 * n_mix, height, width))

        self.logits = l[:, :, :n_mix, :, :]
        self.mu = l[:, :, n_mix:2 * n_mix, :, :]
        log_sig = l[:, :, 2 * n_mix:, :, :]
        self.sig = torch.exp(log_sig) + 1e-2

    def log_p(self, z):
        z = z.unsqueeze(2)

        normalized_samples = (z - self.mu) / self.sig
        log_pz = - 0.5 * normalized_samples * normalized_samples - torch.log(self.sig)
        log_pz -= 0.5 * log2pi

        log_probs = logits_to_log_probs(self.logits)

        res = log_pz + log_probs

        max_l, _ = torch.max(res, dim=2, keepdim=True)
        res = max_l.squeeze(2) + torch.log(torch.sum(torch.exp(res - max_l), dim=2))
        z = z.squeeze(2)

        return res

    def mean(self):
        logit_probs = logits_to_log_probs(self.logits)
        probs = torch.exp(logit_probs)
        res = self.mu * probs
        res = torch.sum(res, dim=2)

        return res

    def most_likely_value(self):
        # technically we arent getting the most_likely_value but
        # the most_likely_value from the most likely mix which is not exactly the same 

        log_probs = logits_to_log_probs(self.logits)
        most_likely_mix = torch.argmax(log_probs, dim=2, keepdim=True)

        x = torch.gather(self.mu, 2, most_likely_mix)
        x = x.squeeze(2)
        return x

    def sample(self):
        selected_mask = torch.distributions.Categorical(logits=self.logits.permute(0, 1, 3, 4, 2))
        selected_mask = selected_mask.sample().unsqueeze(2).to(device)

        x = self.mu + self.sig * torch.randn_like(self.mu).to(device)

        x = torch.gather(x, 2, selected_mask)
        x = x.squeeze(2)
        return x

    @classmethod
    def params_per_subpixel(cls, *args, **kwargs):
        return (Normal.params_per_subpixel() + 1) * args[0]


class DiscLogistic(Distribution):
    def __init__(self, params):
        super().__init__()
        raise NotImplementedError

    @classmethod
    def params_per_subpixel(cls, *args, **kwargs):
        return 2


# this discrete mixture implementation is much simpler than the one from source
class DiscMixLogistic(Distribution):

    def __init__(self, params):
        # assumes c % 9 == 0
        super().__init__()
        batch_size, c, height, width = params.size()
        # note that 3*(2*n_mix) + 3*n_mix == c
        n_mix = c // 9
        l = params.view(batch_size, 3, 3 * n_mix, height, width)

        self.logits = soft_clamp5(l[:, :, :n_mix, :, :])

        # fun fact: mu blows up late in trainning if you dont clamp it. 
        # Maybe it is trying to push the mus to more and more extreme values  
        # when you are trying to reconstruct the pure white/black?
        self.mu = torch.clamp(l[:, :, n_mix:2 * n_mix, :, :], -5, 5)

        log_scales = l[:, :, 2 * n_mix:3 * n_mix, :, :]
        log_scales = torch.clamp(log_scales, min=-7.0)
        self.log_scales = log_scales

    def sample(self):
        # selected_mask = F.softmax(self.logits, tau=t, hard=True, dim=2)
        selected_mask = torch.distributions.Categorical(logits=self.logits.permute(0, 1, 3, 4, 2))
        selected_mask = selected_mask.sample().unsqueeze(2).to(device)
        x = self.mu + torch.exp(self.log_scales) * logistic_distribution.sample(sample_shape=self.mu.shape).to(device)

        x = torch.gather(x, 2, selected_mask)
        x = x.squeeze(2)

        # move to 0-1 range
        x = (x + 1) / 2

        return x

    # stable method adapted from
    # IMPROVING THE PIXELCNN WITH
    # DISCRETIZED LOGISTIC MIXTURE LIKELIHOOD AND
    # OTHER MODIFICATIONS
    def log_p(self, z):
        z = z.unsqueeze(2)
        # move to -1-1 range
        centered_z = 2 * z - 1

        centered_z = (centered_z - self.mu)

        inverted_scale = torch.exp(-self.log_scales)
        plus_in = inverted_scale * (centered_z + 1. / 255)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inverted_scale * (centered_z - 1. / 255)
        cdf_min = torch.sigmoid(min_in)

        log_pz = cdf_plus - cdf_min  # prob (|x-z| <= 1./255) for each subpixel for each part of  mixture

        log_pz = torch.log(torch.clamp(log_pz, min=1e-10))

        # now we take into acount the clipping for the lower and upper parts
        log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in)
        log_inverse_cdf_min = - torch.nn.functional.softplus(min_in)

        log_pz = torch.where(
            z < 5e-4, log_cdf_plus,
            torch.where(z > 1 - 5e-3,
                        log_inverse_cdf_min,
                        log_pz)
        )

        z = z.squeeze(2)

        log_probs = logits_to_log_probs(self.logits)

        # In the next line we have, in index notation:
        # res_b,c,k,h,w =  log(p(|clipped x-z| <= 1./255) | selected = k) + log(p(selected = k))
        res = log_pz + log_probs

        max_l, _ = torch.max(res, dim=2, keepdim=True)
        res = max_l.squeeze(2) + torch.log(torch.sum(torch.exp(res - max_l), dim=2))
        # now we have, in index notation:
        # res_b,c,h,w =  log(sum_k p(|clipped x-z| <= 1./255) and selected = k)
        # = log(p(|clipped x-z| <= 1./255). exactly what we wanted

        return res

    def mean(self):
        log_probs = logits_to_log_probs(self.logits)
        probs = torch.exp(log_probs)
        res = self.mu * probs
        res = torch.sum(res, dim=2)

        res = (res + 1) / 2

        return res

    def most_likely_value(self):
        # technically arent getting the most_likely_value but
        # the most_likely_value from the most likely mix which is not exactly the same 
        log_probs = logits_to_log_probs(self.logits)
        most_likely_mix = torch.argmax(log_probs, dim=2, keepdim=True)
        x = torch.gather(self.mu, 2, most_likely_mix)
        x = x.squeeze(2)

        # move to 0-1 range
        x = (x + 1) / 2
        return x

    @classmethod
    def params_per_subpixel(cls, *args, **kwargs):
        return (DiscLogistic.params_per_subpixel() + 1) * args[0]
