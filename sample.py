import torch
from model_from_config import get_archfile_from_checkpoint, get_model, load_state_from_file
from os.path import exists
from matplotlib import pyplot as plt
import sys
import torchvision

weights_file = sys.argv[1]
config_file = get_archfile_from_checkpoint(weights_file)
if config_file is None:
    raise ValueError("config not found")

model = get_model(config_file)
load_state_from_file(model, weights_file)
model = model.to("cuda:0")


def sample(n1=3, n2=3, batches=1, iterations=20):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[2])
        n = n1 * n2

        for _ in range(iterations):
            model.sample(n, t=t)
        model = model.eval()
        model.mixer.show_temp = True

        for _ in range(batches):
            # most_likely_value
            tensor_images = model.sample(n, t=t, final_distribution_sampling="sample").detach().cpu()
            img_grid = torchvision.utils.make_grid(tensor_images, nrow=n1, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            # torchvision.utils.save_image(img_grid,"hola_"+ str(iterations) +".png")
            plt.show()


iterations = 20 if len(sys.argv) < 4 else int(sys.argv[3])
sample(batches=1, iterations=iterations, n1=3, n2=3)
