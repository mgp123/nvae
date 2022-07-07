import torch
from tqdm import tqdm
from os.path import exists
from data_loaders import get_data_loaders
from kl_scheduler import KLScheduler
from model_from_config import get_model
from model.utils import device
import torchvision
from torch.utils.tensorboard import SummaryWriter


def train():
    model_code_name = "gaussian_mixture"
    weights_file = model_code_name + ".checkpoint"
    config_file = "model_configs/" + model_code_name + ".yaml"
    model = get_model(config_file)

    learning_rate = 1e-2
    weight_decay = 3e-4
    regularization_constant = 5e-2  # prev wa 1e-2 #prev was 5e-2
    kl_constant = 1  # prev was 2.5 #prev was 1
    warmup_epochs = 30
    epochs = 100
    learning_rate_min = 1e-5
    batch_size = 20  # prev was 32

    write_loss_every = 320 * 3 * 3 * 2
    write_loss_every += (-write_loss_every) % batch_size  # to make sure the mod works
    write_images_every = 4378 * 16 * 2  # idem
    write_images_every += (-write_images_every) % batch_size
    checkpoint_every_images = 4378 * 16  # idem
    checkpoint_every_images += (-checkpoint_every_images) % batch_size
    checkpoint_every_epoch = 5

    data_loader_train, data_loader_test = get_data_loaders(batch_size)

    optimizer = torch.optim.Adamax(
        model.parameters(),
        learning_rate,
        weight_decay=weight_decay,
        eps=1e-3)

    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        epochs - warmup_epochs - 1,
        eta_min=learning_rate_min)

    seen_images = 0

    writer = None
    initial_epoch = 0
    initial_seen_images = 0

    if exists(weights_file):
        save = torch.load(weights_file)
        initial_epoch = save["epoch"]
        epochs = save["total_epochs"]
        model = model.to("cuda:0")
        model.load_state_dict(save["state_dict"])
        optimizer.load_state_dict(save["optimizer"])
        optimizer_scheduler.load_state_dict(save["optimizer_scheduler"])

        seen_images = save["seen_images"]
        initial_seen_images = seen_images
        writer = SummaryWriter(log_dir=save["log_dir"])

        print("Loaded checkpoint after seeing " + str(seen_images) + " images")

    else:
        writer = SummaryWriter()
        model = model.to("cuda:0")

    kl_scheduler = KLScheduler(
        kl_warm_steps=warmup_epochs,
        model=model,
        current_step=initial_epoch)

    for epoch in tqdm(range(initial_epoch, epochs), initial=initial_epoch, total=epochs, desc="epoch"):

        for images, _ in tqdm(data_loader_train, leave=False, desc="batch"):
            images = images.to(device)

            optimizer.zero_grad()

            x_distribution, kl_loss = model(images)

            log_p = x_distribution.log_p(images)
            # we sum the independent log_ps for each entry to get the log_p of the whole image
            log_p = torch.sum(log_p, dim=[1, 2, 3])

            kl_loss_balanced = kl_scheduler.warm_up_coeff() * kl_scheduler.balance(kl_loss)
            loss = -log_p + kl_constant * kl_loss_balanced
            loss = torch.mean(loss)

            if regularization_constant != 0:
                loss += regularization_constant * model.regularization_loss()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 200)

            optimizer.step()

            seen_images += batch_size

            if (seen_images - initial_seen_images) % write_loss_every == 0:
                if torch.isnan(loss).any():
                    raise ValueError('Found NaN during training')

                writer.add_scalar("loss/iter", loss.item(), seen_images)
                writer.add_scalar("rec_loss/iter", -torch.mean(log_p).item(), seen_images)
                writer.add_scalar("kl_loss/iter", torch.mean(kl_loss_balanced).item(), seen_images)

                kl_by_split = torch.mean(torch.stack(kl_loss, dim=0), dim=1)

                for i in range(kl_by_split.shape[0]):
                    writer.add_scalar("kl_loss_" + str(i) + "/iter", kl_by_split[i].item(), seen_images)

            if (seen_images - initial_seen_images) % write_images_every == 0:
                with torch.no_grad():
                    # model = model.eval()

                    img = model.sample(2)
                    img_grid = torchvision.utils.make_grid(img)
                    writer.add_image("generated image " + str(seen_images), img_grid)

                    # x = torch.clamp(x_distribution.mu, 0, 1.)
                    # x = x / 2. + 0.5

                    # img = torchvision.utils.make_grid(images)
                    # writer.add_image("original image "  + str(seen_images) , img)

                    x = torch.clamp(x_distribution.sample(), 0, 1.)
                    img = torchvision.utils.make_grid(x)
                    writer.add_image("reconstructed image " + str(seen_images), img)

                    # model = model.train()

            if (seen_images - initial_seen_images) % checkpoint_every_images == 0:
                if torch.isnan(loss).any():
                    raise ValueError('Found NaN during training')

                create_checkpoint(epoch,
                                  epochs,
                                  model,
                                  optimizer,
                                  optimizer_scheduler,
                                  seen_images,
                                  weights_file,
                                  writer)

        if epoch > warmup_epochs:
            optimizer_scheduler.step()
        if epoch % checkpoint_every_epoch == 0:
            create_checkpoint(epoch + 1,
                              epochs,
                              model,
                              optimizer,
                              optimizer_scheduler,
                              seen_images,
                              weights_file,
                              writer)
        kl_scheduler.step()

    writer.flush()
    writer.close()
    torch.save(model.state_dict(), "final_result.model")


def create_checkpoint(epoch, epochs, model, optimizer, optimizer_scheduler, seen_images, weights_file, writer):
    torch.save(
        {"epoch": epoch,
         "state_dict": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "optimizer_scheduler": optimizer_scheduler.state_dict(),
         "seen_images": seen_images,
         "log_dir": writer.log_dir,
         "total_epochs": epochs

         },

        weights_file)


if __name__ == "__main__":
    train()
