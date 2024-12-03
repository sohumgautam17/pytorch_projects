import torch as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import argparse

from dataloader import CIFAR10
from runners import train 
from model import LeNet


def get_args():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--train', type=str, default='store_true', help='Please choose whether it is inference or not')
    parser.add_argument('--epochs', type=int, default=25, help='Please choose how many epochs to train for')
    parser.add_argument('--lr', type=int, default=1e-4, help='Please choose the learning rate for the optimizer')
    parser.add_argument('--model', type=str, default='lenet', help="Please choose a model to train")
    # parser.add_argument('--lr', type=int, default=1e-4, help='Please choose the learning rate for the optimizer')


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
    print(train_data.shape)
    input()

    train_images, train_labels = train_data
    test_images, test_labels = test_data

    train_dataloader = CIFAR10(train_images, train_labels)
    test_dataloader = CIFAR10(test_images, test_labels)


    if args.model == 'lenet':
        model = LeNet()
    optimizer = torch.optim.Adam(model.paramenters, args.lr)
    loss_fn = nn.CrossEntropyLoss()



    if args.train:
        directory = f'./checkpoint/saved_{args.lr}_{args.model}'
        ensure_directory_exists(directory)

        all_epochs = []
        train_losses = []
        for epoch in range(args.epochs):

            model, losses = train(model=model, train_dataloader=train_dataloader,
                                  optimizer=optimizer, loss_fn=loss_fn)
            print(f'Epoch: {epoch} | Loss: {losses}')
            all_epochs.append(epoch)
            train_losses.append(losses)
            
        checkpoint = {
            'model': model.state_dict()
        }

        torch.save(checkpoint, f'./{directory}/checkpoint.chkpt')

    else:
        print('...')

if __name__ == "__main__":
    args = get_args()
    print('#'*40)
    main(args)