from typing import Tuple
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils import data
from os.path import exists


# slightly modified version of the one used in another project 
def get_data_loaders(batch_size) -> Tuple[data.DataLoader, data.DataLoader]:
    if not exists("dataset"):
        raise Exception("No dataset found. You need to put your directory with 128x128 images inside the dataset directory")
    t = transforms.Compose([transforms.Resize(64) ,transforms.ToTensor()])
    dataset = datasets.ImageFolder("dataset", t)
    test_set_size = 4
    test_set, train_set = data.random_split(
        dataset,
        [test_set_size, len(dataset) - test_set_size],
    )

    data_loader_train = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    data_loader_test = data.DataLoader(
        test_set,
        batch_size=test_set_size,
        shuffle=False,
        pin_memory=True
    )

    return data_loader_train, data_loader_test