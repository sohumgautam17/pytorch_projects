import torch as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np

from dataloader import CIFAR10
from runners import train, test 


def get_args():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('train', type=str, default='store_true', help='Please choose whether it is inference or not')

    return parser.parse_args()


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def main(args):
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
    print(len(train_data))

    train_images, train_labels = train_data
    test_images, test_labels = test_data

    train_dataloader = CIFAR10(train_images, train_labels)
    test_dataloader = CIFAR10(test_images, test_labels)

    if args.train:

