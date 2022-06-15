import torch
import torch.nn.functional as F
import torch.distributions


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# taken directly from source 
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.) 

logistic_distribution = torch.distributions.TransformedDistribution(
    torch.distributions.uniform.Uniform(1e-5, 1. - 1e-5), 
    [torch.distributions.SigmoidTransform().inv]
    )

# slightly modified version of Normal class in source
class Normal:
    def __init__(self, mu, log_sig):
        self.mu = soft_clamp5(mu)
        self.sig = torch.exp(log_sig) + 1e-2
    

    def sample(self):
        return torch.randn_like(self.mu)*self.sig + self.mu

    # log probability of observing z_i, each feature of z is done independently. 
    # We ignore constants here
    def log_p(self, z):
        normalized_samples = (z - self.mu) / self.sig
        log_p = - 0.5 * normalized_samples * normalized_samples  - torch.log(self.sig)
        return log_p

# slightly modified version of DiscMixLogistic class in source
class DiscMixLogistic:
    def __init__(self, param):
        
        # it works like this: 
        # for each batch input:
        #   our image distribution is going to be a mixture of num_mix distributions
        #   the probability of each mixture distribution is given by sampling gumbal using the logits parameter
        #       each of these distributions in the mix works like this, for each entry in (3,H,W):
        #             1- sample from logistic distribution
        #             2- scale it acording to its log_scale parameter
        #             3- add its mean
        #       4- do a linear transformation of the channels using this mixture coefs
        
        num_mix = 10
        self.num_mix = num_mix # there are 10 groups of num_mix each
        num_bits = 8
        B, C, H, W = param.size()
        
        self.logits = param[:, :num_mix, :, :]     
        self.logits = self.logits.permute(0, 2, 3,1)

        l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)
        self.means = l[:, :, :num_mix, :, :]                                          
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)   
        self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])  
        self.max_val = 2. ** num_bits - 1

        # include not implemented error until I understand what the log_p does
        raise NotImplementedError
    def log_p(self, z):
        z = 2 * z - 1.0
        z = z.unsqueeze(4)                                                  
        z = z.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)
        
        z0, z1, z2 = torch.chunk(z,3, 1)
        m0, m1, m2 = torch.chunk(self.means,3, 1)
        coeffs0_1, coeffs0_2, coeffs1_2 = torch.chunk(self.coeffs,3, 1)

        # why is z2 not used at all?!
        m1 = coeffs0_1 * z0 + m1            
        m2 = coeffs0_2 * z0 + coeffs1_2 * z1 + m2

        m = torch.cat([m0, m1, m2], dim=1)    
        centered = z - m

        inv_stdv = torch.exp(- self.log_scales)


        # im not really sure whats is the point of whole plus/minus thing
        # maybe it is to prevent the log from blowing up? i dont know

        # this part is taken verbatim from https://github.com/NVlabs/NVAE/blob/9fc1a288fb831c87d93a4e2663bc30ccf9225b29/distributions.py#L131

        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(
            cdf_delta > 1e-5,
            torch.log(torch.clamp(cdf_delta, min=1e-10)),
            log_pdf_mid - 7 )

        log_probs = torch.where(
            z < -0.999, 
            log_cdf_plus, 
            torch.where(
                z > 0.99, 
                log_one_minus_cdf_min,
                log_prob_mid_safe)
                )   


        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logits.permute(0,3,1,2), dim=1)  
        log_probs =  torch.logsumexp(log_probs, dim=1)          
        return  torch.sum(log_probs, dim=[1, 2])

    def sample(self, t=1.):
        selected_mask = F.gumbel_softmax(self.logits, tau=t, hard=True).permute((0, 3, 1,2))
        selected_mask = selected_mask.unsqueeze(1)

        means = torch.sum(self.means * selected_mask, dim=2)
        log_scales = torch.sum(self.log_scales * selected_mask, dim=2)
        coeffs = torch.sum(self.coeffs * selected_mask, dim=2)


        x = means + torch.exp(log_scales) * logistic_distribution.sample(sample_shape=log_scales.size()).to(device)
        x0, x1, x2 = torch.chunk(x,3, 1)
        coeffs0_1, coeffs0_2, coeffs1_2 = torch.chunk(coeffs,3, 1)


        # cant we do this using a convolution with a kernel_size = 1 ?
        x0 = torch.clamp(x0,-1,1)
        x1 = torch.clamp(coeffs0_1 * x0 + x1 , -1, 1)     
        x2 = torch.clamp(coeffs0_2 * x0 + coeffs1_2 * x1 + x2 , -1, 1)     

        x = torch.cat([x0, x1, x2], 1)

        # shift to 0-1 range
        x = x / 2. + 0.5

        return x



# a = torch.rand((3,10*10,2,2))
# d = DiscMixLogistic(a)
# x = d.sample()
# log = d.log_p(x)

# print(log.shape)