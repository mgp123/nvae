import torch
from model.utils import  device

class KLScheduler:
    def __init__(
        self, 
        kl_warm_steps,
        model,
        current_step=0, 
        min_kl_coeff=0 ):

        self.current_step = current_step
        self.min_kl_coeff = min_kl_coeff
        self.kl_warm_steps = kl_warm_steps

        number_of_splits = model.initial_splits_per_scale

        # we are going to multiplie each kl_loss by a weight that depends on the scale it belongs to
        # bigger dimension -> bigger kl_multiplier
        kl_multiplier = []
        for i in range(model.number_of_scales):
            for k in range(number_of_splits):
                kl_multiplier.append(2**(model.number_of_scales-1-i)/number_of_splits)

            number_of_splits = max(model.min_splits, number_of_splits // model.exponential_scaling)
        
        # TODO check if should reverse or not
        kl_multiplier.reverse()
        self.kl_multiplier = torch.FloatTensor(kl_multiplier).unsqueeze(1)
        self.kl_multiplier = self.kl_multiplier/torch.min(self.kl_multiplier)
        self.kl_multiplier = self.kl_multiplier.to(device)

    # for warm up
    def warm_up_coeff(self):
        return max(
            min((self.current_step) / self.kl_warm_steps, 1.0), 
            self.min_kl_coeff)
    
    def balance(self, kl_losses):
        # during warm up you multiplie kl from different scales with different constants 
        if self.current_step < self.kl_warm_steps:
            kl_all = torch.stack(kl_losses, dim=0)

            kl_coeff_i = torch.abs(kl_all)
            # average kl_loss for this group across batches
            kl_coeff_i = torch.mean(kl_coeff_i, dim=1, keepdim=True) + 0.01

            kl_coeff_i = kl_coeff_i / self.kl_multiplier * torch.sum(kl_coeff_i)
            kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)


            return torch.sum(kl_all * kl_coeff_i.detach(), dim=0)
        
        else:
            kl_all = torch.stack(kl_losses, dim=0) # stacks splits kl
            kl_all = torch.mean(kl_all, dim=1) # mean across batches
            kl_all = torch.sum(kl_all) # sums everything up
            return kl_all

    def step(self):
        self.current_step += 1