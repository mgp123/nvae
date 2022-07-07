import torch
from model_from_config import get_model, load_state_from_file
from matplotlib import pyplot as plt
import sys 
import torchvision

weights_file = "saved_weights/final_tiny.model"
config_file = "model_configs/tiny.yaml"

weights_file = "tiny_mixture.checkpoint"
weights_file = "saved_weights/final_tiny_mixture3.model"
config_file = "model_configs/tiny_mixture.yaml"

model = get_model(config_file)
load_state_from_file(model,weights_file)
model = model.to("cuda:0")


def cherry_pick(winners=8):
    global model
    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1]) #0.45
        n1 = 3
        n2 = 3
        n = n1*n2

        for _ in range(1000):
            model.sample(n,t=t)
        model = model.eval()

        selected_images = []

        while len(selected_images) < winners:
            tensor_images = model.sample(n,t=t,final_distribution_sampling="mean").detach().cpu()
            img_grid = torchvision.utils.make_grid(tensor_images, nrow=n1, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            selected = int(input("selected "))
            if selected > 0 and selected <= n1*n2:
                selected_images.append(tensor_images[selected-1])


        img_grid = torchvision.utils.make_grid(selected_images,nrow=3, padding=0)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()
        selected = input("selected ").split(" ")
        selected_images = [selected_images[int(i)-1] for i in selected]
        img_grid = torchvision.utils.make_grid(selected_images, nrow=2,  padding=0)
        # torchvision.utils.save_image(img_grid,"../blog/mypage/images/nvae/pepe.png")
        torchvision.utils.save_image(img_grid,"pepe_mix.png")


cherry_pick(winners=9)