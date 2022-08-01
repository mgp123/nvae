import torch
from model_from_config import get_archfile_from_checkpoint, get_model


checkpoint_file = "saved_weights/final_result_logistic_mixture10latent.model"
config_file = get_archfile_from_checkpoint(checkpoint_file)

model_old = get_model(config_file)
save = torch.load(checkpoint_file)
model_old.load_state_dict(save["state_dict"], strict=False)

config_file = "model_configs/logistic_mixture10latentbiggerspatial.yaml"
model_new = get_model(config_file)

model_new.mixer = model_old.mixer
model_new.encoder_tower = model_old.encoder_tower
model_new.decoder_tower = model_old.decoder_tower
# model_new.decoder_constant = model_old.decoder_constant

model_code_name = "logistic_mixture10latentbiggerspatial"

torch.save(
    {"state_dict": model_new.state_dict(),
    "model_arch": config_file,
    }, 
    
    
    "saved_weights/" + model_code_name + ".model")


print(model_old)
print(model_new)