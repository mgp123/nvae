import torch
import torch.distributions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This regularization does the following for every kernel:
#   - b1 = Wt*W*b_0 and approximates eigen value with ||b1|| / ||b0||

def regularization_conv2d(layer, coefficient=1.):
    # it's too slow
    return 0

    kernels = torch.flatten(layer.weight, end_dim=1)
    sim_kernels = torch.bmm(kernels.permute(0, 2, 1), kernels)
    n = sim_kernels.shape[-1]
    b = 1. * torch.ones(n).to(device)
    b = torch.matmul(sim_kernels, b)
    b = torch.linalg.norm(b, dim=1) / torch.sqrt(torch.tensor(1. * n))
    return coefficient * torch.sum(b)
