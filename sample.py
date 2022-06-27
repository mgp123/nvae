import torch
from model_from_config import get_model
from os.path import exists
from matplotlib import pyplot as plt
import sys 
import torchvision

weights_file = "saved_weights/final_tiny.model"
model = get_model("model_configs/tiny.yaml")

if exists(weights_file):
    if weights_file.split(".")[-1] == "checkpoint":
        save = torch.load(weights_file)
        model = model.to("cuda:0")
        model.load_state_dict(save["state_dict"])
    elif weights_file.split(".")[-1] == "model":
        model = model.to("cuda:0")
        model.load_state_dict(torch.load(weights_file))
    else:
        raise ValueError
else:
    raise FileNotFoundError

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
            tensor_images = model.sample(n,t=t).detach().cpu()
            figure, axis = plt.subplots(n1,n2)

            for i in range(n):
                k1 = i//n2
                k2 = i%n2
                # plt.imshow(tensor_images[i].permute(1, 2, 0))
                # plt.axis('off')
                # plt.tight_layout()
                # plt.show()

                axis[k1,k2].imshow(tensor_images[i].permute(1, 2, 0))
                axis[k1,k2].axis('off')
            plt.tight_layout()
            plt.show()
            selected = int(input("selected "))
            if selected > 0 and selected <= n1*n2:
                selected_images.append(tensor_images[selected-1])


        img_grid = torchvision.utils.make_grid(selected_images, padding=0)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()
        selected = input("selected ").split(" ")
        selected_images = [selected_images[int(i)-1] for i in selected]
        img_grid = torchvision.utils.make_grid(selected_images, nrow=2,  padding=0)
        # torchvision.utils.save_image(img_grid,"../blog/mypage/images/nvae/pepe.png")
        torchvision.utils.save_image(img_grid,"pepe.png")


def sample(n1=3,n2=3,batches=1):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1]) #0.45
        n = n1*n2

        for _ in range(1000):
            model.sample(n,t=t)
        model = model.eval()


        for _ in range(batches):
            tensor_images = model.sample(n,t=t).detach().cpu()
            # figure, axis = plt.subplots(n1,n2)
            img_grid = torchvision.utils.make_grid(tensor_images, nrow=n1, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            plt.show()

# sample(batches=5)

cherry_pick(winners=8)