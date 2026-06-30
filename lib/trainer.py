import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: str | None = None,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

    def _train_loop(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        correct = 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"Training Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        train_acc = 100 * correct / size
        return loss.item(), train_acc

    def _test_loop(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()

        test_loss, correct_preds = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss_fn(pred, y)
                correct_preds += (pred.argmax(1) == y).type(torch.float).sum().item()

            correct_percentage = 100 * correct_preds / size
            test_loss /= num_batches

            print(
                f"Test Error: \nAccuracy: {(correct_percentage):>0.1f}% Avg loss: {test_loss:>8f}\n"
            )
        return test_loss.item(), correct_percentage

    def fit(self, epochs: int):
        print("--------------- Training Loop ---------------\n")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n ---------------------------------------")
            train_loss, train_acc = self._train_loop()
            test_loss, test_acc = self._test_loop()

            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["test_accuracy"].append(test_acc)

    def plot(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, self.history["train_loss"], label="Train loss")
        axes[0].plot(epochs, self.history["test_loss"], label="Test loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, self.history["train_accuracy"], label="Train accuracy")
        axes[1].plot(epochs, self.history["test_accuracy"], label="Test accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
