import torch as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np



# def ensure_directory_exists


train_data = datasets.CIFAR10(
    root='data',
    train= True,
    download=True,
    transform=ToTensor(), # form image PIL format to Torch tensor
    target_transform=None, # for labels
  )

test_data = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None,
  )


# torch
print(train_data)