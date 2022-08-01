import torch
from model_from_config import get_archfile_from_checkpoint, get_model, load_state_from_file
from matplotlib import pyplot as plt
import sys 
import torchvision
from model.utils import  device

# weights_file = "saved_weights/final_tiny.model"
# config_file = "model_configs/tiny.yaml"

# weights_file = "tiny_mixture.checkpoint"
# weights_file = "saved_weights/final_tiny_mixture3.model"
# config_file = "model_configs/tiny_mixture.yaml"

# weights_file = "small_mixture.checkpoint"
# config_file = "model_configs/small_mixture.yaml"

weights_file = "gaussian_mixture.checkpoint"
config_file = "model_configs/gaussian_mixture.yaml"

if len(sys.argv) >= 4:
    weights_file = sys.argv[3]  #"saved_weights/" + sys.argv[3]+".checkpoint"
    tem = get_archfile_from_checkpoint(weights_file)
    if tem is not None:
        config_file = tem

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
                k = 0 if i == 0 else  i - 1 # dummy index for no modification
                temp = torch.clone(latent_zs[k])

                if i!=0:
                    tp = .9 # move it a lot to make the effect visible
                    latent_zs[k] = torch.randn_like(latent_zs[k])*tp

                tensor_image = model.generate_from_latents(latent_zs).detach().cpu()
                tensor_images += [c for c in tensor_image]
                latent_zs[k] = temp

            img_grid = torchvision.utils.make_grid(tensor_images, padding=0,nrow=4)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()



def group_effects(latent_shapes, group, n=3, batches=1, iterations=500):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1]) #0.45
        # t = [t, t,0.2,0.3] 

        for _ in range(iterations):
            model.sample(1,t=t)
        model = model.eval()


        for _ in range(batches):
            latent_zs = []
            for i, latent_shape in enumerate(latent_shapes):
                z = torch.randn((n,latent_shape[0],latent_shape[1],latent_shape[2] ))*t
                z = z.to(device)
                latent_zs.append(z)

            tensor_images = []
            group_size = latent_shapes[group][1]
            tp = 1.7 # move it a lot to make the effect visible
            noise = torch.randn_like(latent_zs[group])*tp

            for i in range(group_size):
                for j in range(group_size):
                    temp = torch.clone(latent_zs[group])

                    latent_zs[group][:,:,i,j] = noise[:,:,i,j]

                    tensor_image = model.generate_from_latents(latent_zs).detach().cpu()
                    tensor_images += [c for c in tensor_image]
                    latent_zs[group] = temp

            img_grid = torchvision.utils.make_grid(tensor_images, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()


# shapes = [(10,16,16),(10,32,32),(10,64,64)]
shapes = [(10,4,4),(10,8,8),(10,16,16),(10,32,32)]
shapes = [(20,8,8),(20,8,8),(20,8,8),(20,16,16),(20,16,16),(20,16,16),(20,32,32),(20,32,32),(20,32,32)]

iterations = 200 if len(sys.argv) < 3 else int(sys.argv[2])

latent_effects(shapes,batches=1, n=1, iterations=iterations)
