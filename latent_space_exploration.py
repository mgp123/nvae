import torch
from model_from_config import get_model, load_state_from_file
from matplotlib import pyplot as plt
import sys 
import torchvision
from model.utils import  device

weights_file = "saved_weights/final_tiny.model"
config_file = "model_configs/tiny.yaml"

weights_file = "tiny_mixture.checkpoint"
weights_file = "saved_weights/final_tiny_mixture3.model"
config_file = "model_configs/tiny_mixture.yaml"

weights_file = "small_mixture.checkpoint"
config_file = "model_configs/small_mixture.yaml"

model = get_model(config_file)
load_state_from_file(model,weights_file)
model = model.to("cuda:0")

def latent_effects(latent_shapes,n=3, batches=1, iterations=500):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1]) #0.45

        for _ in range(iterations):
            model.sample(1,t=t)
        model = model.eval()


        for _ in range(batches):
            latent_zs = []
            for latent_shape in latent_shapes:
                z = torch.randn((n,latent_shape[0],latent_shape[1],latent_shape[2] ))*t
                z = z.to(device)
                latent_zs.append(z)

            tensor_images = []
            for i in range(len(latent_zs)+1):
                k = 0 if i == 0 else len(latent_zs) - i - 1 # dummy index for no modification
                temp = torch.clone(latent_zs[k])

                if i!=0:
                    latent_zs[k] = torch.randn_like(latent_zs[k])*t

                tensor_image = model.generate_from_latents(latent_zs).detach().cpu()
                tensor_images += [c for c in tensor_image]
                latent_zs[k] = temp

            img_grid = torchvision.utils.make_grid(tensor_images, nrow=n, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()


shapes = [(5,8,8),(5,16,16),(5,32,32)]

latent_effects(shapes,batches=5, iterations=200)
