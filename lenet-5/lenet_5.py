import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

batch_size = 32
learning_rate = 1e-3
num_classes = 10
num_epochs = 10

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.07843, std=0.7843),
        ]
    ),
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.07843, std=0.7843),
        ]
    ),
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        C1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(5, 5)), nn.Tanh())
        S2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Tanh(),
        )
        C3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Tanh(),
        )
        S4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Tanh(),
        )
        C5 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.Tanh(),
        )
        F6 = nn.Sequential(nn.Linear(120, 84), nn.Tanh())
        output_layer = nn.Linear(84, 10)

        self.layers = nn.Sequential(
            C1, S2, C3, S4, C5, nn.Flatten(), F6, output_layer
        )

    def forward(self, x):
        output = self.layers(x)
        return output
    

summary(LeNet5(10))
