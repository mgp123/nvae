import time
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from model_from_config import get_model

weights_file = "mymodel_small.model"

model = get_model("model_configs/tiny.yaml")
# save = torch.load(weights_file)
# model.load_state_dict(save["state_dict"])
model = model.to("cuda:0")
"""
model = model.eval()
with torch.no_grad():
    tensor_image = model.sample(1)[0]
    tensor_image = tensor_image.detach().cpu()
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.show()
"""
images = torch.randn((4, 3, 64, 64)).to("cuda:0")
x_distribution, kl_loss = model(images)
