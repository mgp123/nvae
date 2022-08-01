import time
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from model_from_config import get_archfile_from_checkpoint, get_model, load_state_from_file


# dummy mixer jammer
# it may happen that you start to get nans due to the weight normalization division being to small
# specially for the dummy mixer
# this completly overrrides those values

checkpoint_file = "saved_weights/logistic_mixture10latent.checkpoint"
config_file = get_archfile_from_checkpoint(checkpoint_file)

model = get_model(config_file)
save = torch.load(checkpoint_file)
model.load_state_dict(save["state_dict"], strict=False)

weights = model.mixer.encoder_mixer[0].model[1].weight
with torch.no_grad():
  weights = model.mixer.encoder_mixer[0].model[1].weight
  bias = model.mixer.encoder_mixer[0].model[1].bias

  print(torch.linalg.norm(weights,ord=2,dim=[2,3]))
  
  model.mixer.encoder_mixer[0].model[1].weight = torch.nn.Parameter(torch.randn_like(weights))
  model.mixer.encoder_mixer[0].model[1].bias = torch.nn.Parameter(torch.randn_like(bias))


  weights = model.mixer.encoder_mixer[0].model[1].weight

  print(torch.linalg.norm(weights,ord=2,dim=[2,3]))

bias =  model.mixer.encoder_mixer[0].model[1].bias


