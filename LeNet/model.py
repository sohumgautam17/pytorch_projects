import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms 

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cn1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.cn2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.flatten = nn.Flatten()

        self.ffn = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 10)
        )
    
    def forward(self, x):
        x = self.cn1(x)
        x = self.cn2(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.ffn(x)
        return x

inp = torch.rand((32, 32, 3), dtype=torch.float32)   
print(inp.shape)

inp = inp.permute(2, 0, 1)
inp = inp.unsqueeze(0)
print(inp.shape)

model = LeNet()
output = model(inp)
print(f'Output Shape: {output.shape}')

