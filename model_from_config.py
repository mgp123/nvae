import torch
import yaml
from os.path import exists

from model.autoencoder import Autoencoder


def read_config(yaml_path):
    res = {}
    with open(yaml_path, "r") as stream:
        res = (yaml.safe_load(stream))
    return res


def get_model(yaml_path) -> Autoencoder:
    conf = read_config(yaml_path)
    res = Autoencoder(
        conf["channel_towers"],
        conf["number_of_scales"],
        conf["initial_splits_per_scale"],
        conf["latent_size"],
        conf["input_dimension"],
        conf["num_flows"],
        conf.get("num_blocks_prepost", 1),
        conf.get("num_cells_per_block_prepost", 2),
        conf.get("cells_per_split_enc", 2),
        conf.get("cells_per_input_dec", 1),
        conf.get("channel_multiplier", 2),
        conf.get("exponential_scaling", 1),
        conf.get("min_splits", 1),
        conf.get("sampling_method", "gaussian"),
        conf.get("n_mix", 3),
        conf.get("fixed_flows", False)
    )
    return res


def get_archfile_from_checkpoint(checkpoint_file):
    if checkpoint_file is not None and exists(checkpoint_file):
        conf = torch.load(checkpoint_file)
        return conf.get("model_arch", None)
    return None

def get_training_params(yaml_path=None, checkpoint_file=None):
    if yaml_path is None and checkpoint_file is None:
        raise ValueError

    if checkpoint_file is not None and exists(checkpoint_file):
        conf = torch.load(checkpoint_file)
        yaml_path = conf.get("model_arch", yaml_path)

    if yaml_path is None or not exists(yaml_path):
        raise ValueError

    conf = read_config(yaml_path)
    training_parameters = {}
    if "training_parameters" in conf:
        training_parameters = conf["training_parameters"]

    training_parameters["learning_rate"] = training_parameters.get("learning_rate", 1e-2)
    training_parameters["regularization_constant"] = training_parameters.get("regularization_constant", 5e-2)
    training_parameters["kl_constant"] = training_parameters.get("kl_constant", 1)

    training_parameters["warmup_epochs"] = training_parameters.get("warmup_epochs", 30)

    training_parameters["batch_size"] = training_parameters.get("batch_size", 20)
    batch_size = training_parameters["batch_size"]

    training_parameters["write_reconstruction"] = training_parameters.get("write_reconstruction", True)

    training_parameters["images_per_checkpoint"] = training_parameters.get("images_per_checkpoint", None)
    if training_parameters["images_per_checkpoint"] is not None:
        training_parameters["images_per_checkpoint"] += (-training_parameters["images_per_checkpoint"]) % batch_size

    training_parameters["epochs_per_checkpoint"] = training_parameters.get("epochs_per_checkpoint", 2)
    training_parameters["epochs"] = training_parameters.get("epochs", -1)

    training_parameters["gradient_clipping"] = training_parameters.get("gradient_clipping", None)
    training_parameters["half_precision"] = training_parameters.get("half_precision", False)
    training_parameters["use_tensor_checkpoints"] = training_parameters.get("use_tensor_checkpoints", False)


    return training_parameters


def load_state_from_file(model, weights_file):
    type_format = weights_file.split(".")[-1]
    if exists(weights_file):
        if  type_format == "checkpoint" or type_format == "model":
            save = torch.load(weights_file)
            model.load_state_dict(save["state_dict"], strict=False)
        else:
            raise ValueError("file must be .model or .checkpoint")
    else:
        raise FileNotFoundError
