import torch
from model_from_config import get_archfile_from_checkpoint, get_model, load_state_from_file
from os.path import exists
from matplotlib import pyplot as plt
import sys 
import torchvision

# weights_file = "saved_weights/final_tiny.model"
# config_file = "model_configs/tiny.yaml"

# weights_file = "tiny_mixture.checkpoint"
# weights_file = "saved_weights/final_tiny_mixture3.model"
# config_file = "model_configs/tiny_mixture.yaml"

# weights_file = "small_mixture.checkpoint"
# config_file = "model_configs/small_mixture.yaml"

# weights_file = "small_mixture2.checkpoint"
# config_file = "model_configs/small_mixture2.yaml"

weights_file = "gaussian_mixture.checkpoint"
config_file = "model_configs/gaussian_mixture.yaml"

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
# print("fixed flows", model.mixer.fixed_flows)
# model.mixer.fixed_flows = True

def sample(n1=3,n2=3,batches=1, iterations=20):
    global model

    with torch.no_grad():
        # readjusting batchnorm
        t = float(sys.argv[1]) #0.45
        # t = [t, t,0.3,0.1] # good combination for 0.3-0.45 temps
        # t = [t, t,0.2,0.3] 
        t = [t]*10
        t[4] = 0.4
        t[5] = t[4]
        t[6] = t[5]
        #t[7] = 0.9
        #t[8] = t[7]
        #t[9] = t[8]
        
        n = n1*n2

        for _ in range(iterations):
            model.sample(n,t=t)
        model = model.eval()
        model.mixer.show_temp = True


        for _ in range(batches):
            # most_likely_value
            tensor_images = model.sample(n,t=t,final_distribution_sampling="sample").detach().cpu()
            img_grid = torchvision.utils.make_grid(tensor_images, nrow=n1, padding=0)
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            # torchvision.utils.save_image(img_grid,"hola_"+ str(iterations) +".png")
            plt.show()


iterations = 20 if len(sys.argv) < 3 else int(sys.argv[2])
sample(batches=6, iterations=iterations, n1=3,n2=3)

