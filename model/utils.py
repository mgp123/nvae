import torch
import torch.distributions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This regularization does the following for every kernel:
#   - b1 = Wt*W*b_0 and approximates eigen value with ||b1|| / ||b0||

def regularization_conv2d(layer, coefficient=1):
    W = torch.flatten(layer.weight,start_dim=1)
    # W = layer.weight.view(layer.weight.shape[0],-1)


    v0 = torch.randn(W.shape[-1],device=device)
    v0 = torch.nn.functional.normalize(v0, eps=1e-3, dim=0)
    
    u = torch.mv(W,v0)
    # u = torch.nn.functional.normalize(u, eps=1e-3,dim=0)

    v1 = torch.mv(W.t(), u)  # Wt W v0
    v1 = torch.nn.functional.normalize(v1, eps=1e-3,dim=0)

    # l = torch.matmul(u, torch.mv(W,v1))
    # return l

    l = torch.matmul(W,v1)
    l = torch.sum(l**2)
    return l*coefficient


    s = torch.norm(layer.weight,dim=[2,3])
    return torch.sum(s)* coefficient

    # it's too slow
    kernels = torch.flatten(layer.weight, end_dim=1)
    sim_kernels = torch.bmm(kernels.permute(0, 2, 1), kernels)
    n = sim_kernels.shape[-1]
    b = 1. * torch.ones(n).to(device)
    b = torch.matmul(sim_kernels, b)
    b = torch.linalg.norm(b, dim=1) / torch.sqrt(torch.tensor(1. * n))
    return coefficient * torch.sum(b)
