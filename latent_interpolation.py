import torch
from model_from_config import get_archfile_from_checkpoint, get_model, load_state_from_file
from matplotlib import pyplot as plt
import sys
import torchvision
from model.utils import device

weights_file = sys.argv[1]
config_file = get_archfile_from_checkpoint(weights_file)
if config_file is None:
    raise ValueError("config not found")

model = get_model(config_file)
load_state_from_file(model, weights_file)
model.eval()
model = model.to("cuda:0")

path1 = sys.argv[2]
path2 = sys.argv[3]
n = 1 if len(sys.argv) < 5 else  int(sys.argv[4])

with torch.no_grad():
    codings = []
    original_images = []

    for path in [path1,path2]:
        image = torchvision.io.read_image(path)/255
        image = torchvision.transforms.Resize(model.input_dimension)(image)

        original_image = image

        original_images.append(original_image)
        image = image.to(device).unsqueeze(0)

        zs = model.encode(image)
        codings.append(zs)


    factor_multiplier = 1/(n+1)

    images = []
    for i in range(1,n+1):

        alpha = i*factor_multiplier
        zs = [ a*(1-alpha) + b*alpha  for a, b in zip(*codings)]

        image = model.decode(zs)
        image = image.detach().cpu()
        image = image.squeeze(0)
        images.append(image)

    images = [original_images[0] ] + images + [original_images[1]]

    img_grid = torchvision.utils.make_grid(images,  padding=0)
    # torchvision.utils.save_image(img_grid,"mix"+".png")

    plt.imshow(img_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
