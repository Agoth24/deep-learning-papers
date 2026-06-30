import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from lib.trainer import Trainer
import os


batch_size = 32
learning_rate = 1e-3
num_classes = 10
num_epochs = 10

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

train_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH,
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
    root=DATA_PATH,
    train=False,
    download=True,
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

        self.layers = nn.Sequential(C1, S2, C3, S4, C5, nn.Flatten(), F6, output_layer)

    def forward(self, x):
        output = self.layers(x)
        return output


summary(LeNet5(num_classes))


model = LeNet5(num_classes).to(device)
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (images, labels) in enumerate(dataloader):
#         images, labels = images.to(device), labels.to(device)

#         # forward pass
#         pred = model(images)
#         loss = loss_fn(pred, labels)

#         # backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(images)
#             print(f"Training Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


# def test_loop(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()

#     test_loss, correct_preds = 0, 0
#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)

#             pred = model(images)
#             test_loss += loss_fn(pred, labels)
#             correct_preds += (pred.argmax(1) == labels).type(torch.float).sum().item()

#         test_loss /= num_batches
#         correct_preds /= size
#         print(
#             f"Test Error: \nAccuracy: {(correct_preds*100):>0.1f}% Avg loss: {test_loss:>8f}\n"
#         )

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch}\n ---------------------------------------")
#     train_loop(train_dataloader, model=model, loss_fn=loss_criterion, optimizer=optimizer)
#     test_loop(test_dataloader, model=model, loss_fn=loss_criterion)


# os.makedirs("checkpoints", exist_ok=True)
# torch.save(model.state_dict(), "./checkpoints/lenet5_baseline.pth")

trainer = Trainer(train_dataloader, test_dataloader, model, loss_criterion, optimizer, device)
trainer.fit(epochs=10)
trainer.plot()