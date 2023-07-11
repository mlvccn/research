import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import dataset, transform

data_dir='./dataset'
bs = 8
mean = [ 58.6573,  65.9755,  56.4990, 100.8296]
std = [60.2980, 59.0457, 58.0989, 47.1224]
train_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)]
)

train_data = dataset.SemData(split='train', data_root=data_dir, transform=train_transform)

train_loader  = DataLoader(
    dataset     = train_data,
    batch_size  = bs,
    shuffle     = True,
    num_workers = 4,
    pin_memory  = True,
    drop_last   = False
)

data, label, _, _ = next(iter(train_loader))
for i in range(len(data)):
    print(data[i].mean(), data[i].std())

# https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

mean, std = get_mean_and_std(train_loader)
print(mean, std)

def mean_std_for_loader(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for data, _, _, _ in loader:
        this_batch_size = data.size()[0]
        weight = this_batch_size / loader.batch_size
        channels_sum += weight*torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += weight*torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += weight
    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

# mean, std = mean_std_for_loader(train_loader)
# print(mean, std)

# the result is below
# mean = [0.2216, 0.2587, 0.2300, 0.3954]
# std = [0.2278, 0.2316, 0.2365, 0.1848]