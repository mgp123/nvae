import torch
from model_from_config import get_model, load_state_from_file
from os.path import exists
from matplotlib import pyplot as plt
import sys 
import torchvision

weights_file = "saved_weights/final_tiny.model"
config_file = "model_configs/tiny.yaml"

weights_file = "tiny_mixture.checkpoint"
weights_file = "saved_weights/final_tiny_mixture3.model"
config_file = "model_configs/tiny_mixture.yaml"

weights_file = "small_mixture.checkpoint"
config_file = "model_configs/small_mixture.yaml"

weights_file = "small_mixture2.checkpoint"
config_file = "model_configs/small_mixture2.yaml"

model = get_model(config_file)
load_state_from_file(model,weights_file)
model = model.to("cuda:0")

def sample(n1=3,n2=3,batches=1, iterations=500):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1]) #0.45
        n = n1*n2

        for _ in range(iterations):
            model.sample(n,t=t)
        model = model.eval()


        for _ in range(batches):
            tensor_images = model.sample(n,t=t,final_distribution_sampling="most_likely_value").detach().cpu()
            img_grid = torchvision.utils.make_grid(tensor_images, nrow=n1, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()


iterations = 200 if len(sys.argv) < 3 else int(sys.argv[2])
sample(batches=8, iterations=iterations)

