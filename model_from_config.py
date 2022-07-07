import torch
import yaml
from os.path import exists

from model.autoencoder import Autoencoder

def read_config(yaml_path):
    res = {}
    with open(yaml_path, "r") as stream:
        res = (yaml.safe_load(stream))
    return res

def get_model(yaml_path):
    conf = read_config(yaml_path)
    res = Autoencoder(
            conf["channel_towers"],
            conf["number_of_scales"],
            conf["initial_splits_per_scale"],
            conf["latent_size"],
            conf["input_dimension"],
            conf["num_flows"],
            conf.get("num_blocks_prepost",1),
            conf.get("num_cells_per_block_prepost",2),
            conf.get("cells_per_split_enc",2),
            conf.get("cells_per_input_dec",1),
            conf.get("channel_multiplier",2),
            conf.get("exponential_scaling",1),
            conf.get("min_splits",1),
            conf.get("sampling_method","gaussian"),
            conf.get("n_mix", 3),
            conf.get("fixed_flows", False)

    )
    return res


def load_state_from_file(model, weights_file):
    if exists(weights_file):
        if weights_file.split(".")[-1] == "checkpoint":
            save = torch.load(weights_file)
            model.load_state_dict(save["state_dict"])
        elif weights_file.split(".")[-1] == "model":
            model.load_state_dict(torch.load(weights_file))
        else:
            raise ValueError("file must be .model or .checkpoint")
    else:
        raise FileNotFoundError