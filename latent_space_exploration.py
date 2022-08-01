import torch

from model.autoencoder import Autoencoder
from model_from_config import get_archfile_from_checkpoint, get_model, load_state_from_file
from matplotlib import pyplot as plt
import sys
import torchvision
from model.utils import device

# weights_file = "saved_weights/final_tiny.model"
# config_file = "model_configs/tiny.yaml"

# weights_file = "tiny_mixture.checkpoint"
# weights_file = "saved_weights/final_tiny_mixture3.model"
# config_file = "model_configs/tiny_mixture.yaml"

# weights_file = "small_mixture.checkpoint"
# config_file = "model_configs/small_mixture.yaml"

weights_file = "gaussian_mixture.checkpoint"
config_file = "model_configs/gaussian_mixture.yaml"

if len(sys.argv) >= 3:
    weights_file = sys.argv[2]
    tem = get_archfile_from_checkpoint(weights_file)
    if tem is not None:
        config_file = tem

model = get_model(config_file)
load_state_from_file(model, weights_file)
model = model.to("cuda:0")


def latent_effects(latent_shapes, n=3, batches=1, iterations=500):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1])  # 0.45

        for _ in range(iterations):
            model.sample(1, t=t)
        model = model.eval()

        for _ in range(batches):
            latent_zs = []
            for latent_shape in latent_shapes:
                z = torch.randn((n, latent_shape[0], latent_shape[1], latent_shape[2])) * t
                z = z.to(device)
                latent_zs.append(z)

            tensor_images = []
            for i in range(len(latent_zs) + 1):
                k = 0 if i == 0 else i - 1  # dummy index for no modification
                temp = torch.clone(latent_zs[k])

                if i != 0:
                    tp = .8  # move it a lot to make the effect visible
                    latent_zs[k] = torch.randn_like(latent_zs[k]) * tp

                tensor_image = model.generate_from_latents(latent_zs).detach().cpu()
                tensor_images += [c for c in tensor_image]
                latent_zs[k] = temp

            img_grid = torchvision.utils.make_grid(tensor_images, padding=0, nrow=4)

            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            save_name = config_file.split("/")[-1].split(".")[0]
            # torchvision.utils.save_image(img_grid, "images/latent_space_exploration_" + save_name + ".png")


def group_effects(latent_shapes, group, n=3, batches=1, iterations=500):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1])  # 0.45
        # t = [t, t,0.2,0.3] 

        for _ in range(iterations):
            model.sample(1, t=t)
        model = model.eval()

        for _ in range(batches):
            latent_zs = []
            for i, latent_shape in enumerate(latent_shapes):
                z = torch.randn((n, latent_shape[0], latent_shape[1], latent_shape[2])) * t
                z = z.to(device)
                latent_zs.append(z)

            tensor_images = []
            group_size = latent_shapes[group][1]
            tp = 0.7  # move it a lot to make the effect visible
            noise = torch.randn_like(latent_zs[group]) * tp

            for i in range(group_size):
                for j in range(group_size):
                    temp = torch.clone(latent_zs[group])

                    latent_zs[group][:, :, i, j] = noise[:, :, i, j]

                    tensor_image = model.generate_from_latents(latent_zs).detach().cpu()
                    tensor_images += [c for c in tensor_image]
                    latent_zs[group] = temp

            img_grid = torchvision.utils.make_grid(tensor_images, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()


def get_latent_shapes(m: Autoencoder):
    res = []
    s = m.input_dimension // (2 ** m.num_blocks_prepost)
    i_in_scale = m.initial_splits_per_scale
    for _ in range(m.number_of_scales):
        for _ in range(i_in_scale):
            res.append((m.latent_size, s, s))
        i_in_scale = max(i_in_scale // m.exponential_scaling, m.min_splits)

        s = s // 2
    res.reverse()

    return res


shapes = get_latent_shapes(model)
iterations = 20 if len(sys.argv) < 4 else int(sys.argv[3])

latent_effects(shapes, batches=1, n=1, iterations=iterations)
