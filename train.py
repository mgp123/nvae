import sys
import torch
from tqdm import tqdm
from os.path import exists
from data_loaders import get_data_loaders
from dummyWith import dummyWith
from kl_scheduler import KLScheduler
from model_from_config import get_archfile_from_checkpoint, get_model, get_training_params
from model.utils import device
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.utils.checkpoint as checkpoint

torch.backends.cudnn.benchmark = True

def train():
    model_code_name = "logistic_mixture10latent"
    model_code_name = sys.argv[1]
    checkpoint_file = "saved_weights/" + model_code_name + ".checkpoint"
    config_file = get_archfile_from_checkpoint(checkpoint_file)

    if config_file is None:
        config_file = "model_configs/" + model_code_name + ".yaml"

    model = get_model(config_file)

    learning_rate = 1e-2
    weight_decay = 3e-4
    regularization_constant = 5e-2  # prev wa 1e-2 #prev was 5e-2
    kl_constant = 1  # prev was 2.5 #prev was 1
    warmup_epochs = 30
    epochs = 100
    learning_rate_min = 1e-5
    batch_size = 40  # prev was 20  # prev was 32

    write_reconstruction = True
    save_samples_during_training = False
    # images_per_checkpoint = 4378 * 16 * 1000  # idem
    # images_per_checkpoint += (-images_per_checkpoint) % batch_size
    images_per_checkpoint = None

    epochs_per_checkpoint = 2

    training_parameters = get_training_params(config_file, checkpoint_file)
    learning_rate = training_parameters.get("learning_rate")
    regularization_constant = training_parameters.get("regularization_constant", )
    kl_constant = training_parameters.get("kl_constant")
    warmup_epochs = training_parameters.get("warmup_epochs")
    epochs = training_parameters.get("epochs")
    batch_size = training_parameters.get("batch_size")
    write_reconstruction = training_parameters.get("write_reconstruction")
    images_per_checkpoint = training_parameters.get("images_per_checkpoint")
    epochs_per_checkpoint = training_parameters.get("epochs_per_checkpoint")
    gradient_clipping = training_parameters.get("gradient_clipping")
    half_precision = training_parameters.get("half_precision")
    use_tensor_checkpoints = training_parameters.get("use_tensor_checkpoints")


    write_loss_every = 320 * 3 * 3 * 2
    write_loss_every += (-write_loss_every) % batch_size  # to make sure the mod works
    write_images_every = 4375 * 16 * 2  # idem
    write_images_every += (-write_images_every) % batch_size

    precision_opener = torch.cuda.amp.autocast if half_precision else dummyWith
    model.set_use_tensor_checkpoints(use_tensor_checkpoints)


    data_loader_train, data_loader_test = get_data_loaders(batch_size, model.input_dimension)

    optimizer = torch.optim.Adamax(
        model.parameters(),
        learning_rate,
        weight_decay=weight_decay,
        eps=1e-3)

    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        epochs - warmup_epochs - 1,
        eta_min=learning_rate_min)

    # torch.nn.utils.remove_weight_norm(model.mixer.encoder_mixer[0].model[1])

    seen_images = 0

    writer = None
    initial_epoch = 0
    initial_seen_images = 0
    scaler = torch.cuda.amp.GradScaler()

    if exists(checkpoint_file):
        save = torch.load(checkpoint_file)
        initial_epoch = save["epoch"]
        model = model.to("cuda:0")
        model.load_state_dict(save["state_dict"], strict=False)

        optimizer.load_state_dict(save["optimizer"])
        optimizer_scheduler.load_state_dict(save["optimizer_scheduler"])

        seen_images = save["seen_images"]
        initial_seen_images = seen_images
        writer = SummaryWriter(log_dir=save["log_dir"])

        del save

        print("Loaded checkpoint after " + str(initial_epoch) + " epochs")

    else:
        writer = SummaryWriter(log_dir="runs/" + model_code_name)
        model = model.to("cuda:0")




    kl_scheduler = KLScheduler(
        kl_warm_steps=warmup_epochs,
        model=model,
        current_step=initial_epoch)

    last_loss = 0 # only used to check for nans before checkpoints 
    last_reg_loss = 0
    reg_loss_threshold = 2.5e+4

    for epoch in tqdm(range(initial_epoch, epochs), initial=initial_epoch, total=epochs, desc="epoch"):

        for images, _ in tqdm(data_loader_train, leave=False, desc="batch"):
            images = images.to(device)

            optimizer.zero_grad()

            with precision_opener():
                
                x_distribution, kl_loss = model(images)

                log_p = x_distribution.log_p(images)
                # we sum the independent log_ps for each entry to get the log_p of the whole image
                log_p = torch.sum(log_p, dim=[1, 2, 3])

                kl_loss_balanced = kl_scheduler.warm_up_coeff() * kl_scheduler.balance(kl_loss)
                loss = -log_p + kl_constant * kl_loss_balanced
                loss = torch.mean(loss)

                if regularization_constant != 0:
                    regularization_loss = model.regularization_loss()
                    loss += regularization_constant * regularization_loss
                else:
                    regularization_loss = 0

            if half_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            if half_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            seen_images += batch_size
            last_loss = loss.item()
            last_reg_loss = regularization_loss.item()

            if (write_loss_every is not None) and (seen_images - initial_seen_images) % write_loss_every == 0:
                if torch.isnan(loss).any():
                    raise ValueError('Found NaN during training')

                writer.add_scalar("loss/iter", loss.item(), seen_images)
                writer.add_scalar("rec_loss/iter", -torch.mean(log_p).item(), seen_images)
                writer.add_scalar("kl_loss/iter", torch.mean(kl_loss_balanced).item(), seen_images)
                
                if regularization_constant != 0:
                    writer.add_scalar("reg_loss/iter", regularization_loss.item(), seen_images)

                kl_by_split = torch.mean(torch.stack(kl_loss, dim=0), dim=1)

                for i in range(kl_by_split.shape[0]):
                    writer.add_scalar("kl_loss_" + str(i) + "/iter", kl_by_split[i].item(), seen_images)

            if write_reconstruction and (write_images_every is not None) and (seen_images - initial_seen_images) % write_images_every == 0:
                with torch.no_grad():
                    # model = model.eval()

                    x = torch.clamp(x_distribution.sample(), 0, 1.)
                    img = torchvision.utils.make_grid(x)
                    writer.add_image("reconstructed image", img, global_step=seen_images)

                    # model = model.train()

            if (images_per_checkpoint is not None)  and (seen_images - initial_seen_images) % images_per_checkpoint == 0:
                if torch.isnan(loss).any():
                    raise ValueError('Found NaN during training')

                create_checkpoint(epoch,
                                  epochs,
                                  model,
                                  optimizer,
                                  optimizer_scheduler,
                                  seen_images,
                                  checkpoint_file,
                                  writer,
                                  config_file)

        if epoch > warmup_epochs:
            optimizer_scheduler.step()
        if epoch % epochs_per_checkpoint == 0:

            if last_loss != last_loss:
                raise ValueError('Found NaN during training')
            if reg_loss_threshold is not None and last_reg_loss > reg_loss_threshold:
                raise ValueError('Regularization loss spiked during training')

            img = None
            with torch.no_grad():
                img = model.sample(2)
                
            img_grid = torchvision.utils.make_grid(img)
            writer.add_image("generated image", img_grid, global_step=seen_images)


            create_checkpoint(epoch + 1,
                              epochs,
                              model,
                              optimizer,
                              optimizer_scheduler,
                              seen_images,
                              checkpoint_file,
                              writer,
                              config_file)
            
            if save_samples_during_training:
                with torch.no_grad():
                    for p in [0.2,0.4,0.6]:
                        img = model.sample(9,t=p)
                        img_grid = torchvision.utils.make_grid(img, nrow=3, padding=0)
                        torchvision.utils.save_image(img_grid,"local/samples/trained_"+ str(p) +"_" + str(epoch) +".png")
                

        kl_scheduler.step()

    writer.flush()
    writer.close()
    torch.save(
        {"state_dict": model.state_dict(),
        "model_arch": config_file,
        }, 
        
        
        "saved_weights/final_result_" + model_code_name + ".model")


def create_checkpoint(epoch, epochs, model, optimizer, optimizer_scheduler, seen_images, checkpoint_file, writer,
                      config_file):
    torch.save(
        {"epoch": epoch,
         "state_dict": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "optimizer_scheduler": optimizer_scheduler.state_dict(),
         "seen_images": seen_images,
         "log_dir": writer.log_dir,
         "total_epochs": epochs,
         "model_arch": config_file,
         },

        checkpoint_file)


if __name__ == "__main__":
    train()
