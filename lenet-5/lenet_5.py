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
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()
    else "cpu"
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.07843, std=0.7843)
    ])
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.07843, std=0.7843)
    ])
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


