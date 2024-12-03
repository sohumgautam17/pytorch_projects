import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    ''''''

    def __init__(self, images, labels):
        self.images = images 
        self.labels = labels


    def __len__():
        return len(self.images)

    def __getitem__():
        image = self.images[0]
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        label = self.label[0]


