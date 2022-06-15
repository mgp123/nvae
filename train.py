import torch
from tqdm import tqdm
from os.path import exists
from data_loaders import get_data_loaders
from kl_scheduler import KLScheduler
from model_from_config import get_model
from model.utils import device
import torchvision
from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import GradScaler


if __name__ == "__main__":

    weights_file = "mymodel_new.model"
    model = get_model("model_configs/celeb64.yaml")

    learning_rate = 1e-2
    weight_decay = 3e-4
    warmup_epochs = 10
    epochs = 10
    learning_rate_min =1e-5
    batch_size = 4
    write_loss_every = 320 # make sure that write_loss_every%batch_size == 0
    write_images_every = 9600*8 # idem
    checkpoint_every = 9600*2 # idem
    data_loader_train, data_loader_test = get_data_loaders(batch_size)
    
    optimizer = torch.optim.Adamax(
        model.parameters(),
        learning_rate,
        weight_decay=weight_decay,
        eps=1e-3 )

    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        float(epochs - warmup_epochs - 1), 
        eta_min=learning_rate_min)

    seen_images = 0

    writer = None
    initial_epoch = 0

    if exists(weights_file):
        save = torch.load(weights_file)
        initial_epoch = save["epoch"]
        epochs = save["total_epochs"]
        model = model.to("cuda:0")
        model.load_state_dict(save["state_dict"])
        optimizer.load_state_dict(save["optimizer"])
        optimizer_scheduler.load_state_dict(save["optimizer_scheduler"])
        
        seen_images = save["seen_images"]
        writer = SummaryWriter(log_dir=save["log_dir"])
        
        print("Loaded checkpoint after seeing " + str(seen_images) + " images")

    else:
        writer = SummaryWriter()
        model = model.to("cuda:0")


    kl_scheduler = KLScheduler(
        kl_warm_steps=20,
        model=model,
        current_step=initial_epoch)


    for epoch in tqdm(range(initial_epoch, epochs), initial=initial_epoch, total=epochs, desc="epoch"):

        for images, _ in tqdm(data_loader_train, leave=False, desc="batch"):
            images = images.to(device)
            optimizer.zero_grad()

            
            x_distribution, kl_loss = model(images)
            log_p = x_distribution.log_p(images)
            # we sum the independent log_ps for each entry to get the log_p of the whole image
            log_p = torch.sum(log_p, dim=[1,2,3]) 

            kl_loss = kl_scheduler.warm_up_coeff()*kl_scheduler.balance(kl_loss)
            loss = -log_p + kl_loss
            loss = torch.mean(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

            optimizer.step()


            seen_images += batch_size

            if seen_images % write_loss_every == 0:
                if torch.isnan(loss).any():
                    raise ValueError('Found NaN during training')

                writer.add_scalar("loss/iter", loss.item(), seen_images)
                writer.add_scalar("rec_loss/iter", -torch.mean(log_p).item(), seen_images)
                writer.add_scalar("kl_loss/iter", torch.mean(kl_loss).item(), seen_images)

            if seen_images % write_images_every == 0:
                with torch.no_grad():

                    img = model.sample(2)
                    img_grid = torchvision.utils.make_grid(img)
                    writer.add_image("generated image " + str(seen_images), img_grid)

                    x = torch.clamp(x_distribution.mu, 0, 1.) 
                    # x = x / 2. + 0.5

                    # img = torchvision.utils.make_grid(images)
                    # writer.add_image("original image "  + str(seen_images) , img)

                    img = torchvision.utils.make_grid(x)
                    writer.add_image("reconstructed image "  + str(seen_images), img)

            if seen_images % checkpoint_every == 0:
                if torch.isnan(loss).any():
                    raise ValueError('Found NaN during training')

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


        if epoch > warmup_epochs:
            optimizer_scheduler.step()    

        kl_scheduler.step()

    writer.flush()
    writer.close()
