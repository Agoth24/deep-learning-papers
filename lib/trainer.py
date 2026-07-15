import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
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
        scheduler: lr_scheduler.LRScheduler | None = None,
        step_scheduler_per_batch: bool = False,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps" if (hasattr(torch, "mps") and torch.mps.is_available()) else "cpu"
            )
        )
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_scheduler_per_batch = step_scheduler_per_batch
        self.loss_fn = loss_fn

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

    def _train_loop(self):
        train_set_size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        self.model.train()

        train_loss, correct_preds = 0, 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            if self.scheduler and self.step_scheduler_per_batch:
                self.scheduler.step()

            train_loss += loss.item()
            correct_preds += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 100 == 0:
                current = batch * len(X)
                print(
                    f"Training Loss: {loss.item():>7f} [{current:>5d}/{train_set_size:>5d}]"
                )
        train_loss /= num_batches
        train_acc = 100 * correct_preds / train_set_size
        return train_loss, train_acc

    def _test_loop(self):
        test_set_size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()

        test_loss, correct_preds = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                test_loss += loss.item()
                correct_preds += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            test_acc = 100 * correct_preds / test_set_size

            print(
                f"Test Error: \nAccuracy: {(test_acc):>0.1f}% Avg loss: {test_loss:>8f}\n"
            )
        return test_loss, test_acc

    def fit(
        self,
        epochs: int,
        start_epoch: int = 0,
        save_frequency: int | None = None,
        save_path: str = "checkpoint.pth",
    ):
        print("--------------- Training Loop ---------------\n")
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch + 1}\n---------------------------------------")
            train_loss, train_acc = self._train_loop()
            if self.scheduler and not self.step_scheduler_per_batch:
                self.scheduler.step()
            test_loss, test_acc = self._test_loop()

            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["test_accuracy"].append(test_acc)

            if save_frequency and (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(save_path, epoch)

    def save_checkpoint(self, path: str, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # automated warning messages for loading w/ schedulers at checkpoints
        if checkpoint["scheduler_state_dict"] is not None and self.scheduler is None:
            print(
                "Warning: checkpoint has scheduler state but this Trainer has no scheduler — skipping."
            )
        elif self.scheduler is not None and checkpoint["scheduler_state_dict"] is None:
            print(
                "Warning: this Trainer has a scheduler but checkpoint has none — scheduler state not restored."
            )
        elif (
            self.scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.history = checkpoint["history"]
        return checkpoint["epoch"]

    def resume(
        self,
        path: str,
        epochs: int,
        save_frequency: int | None = None,
        save_path: str = "checkpoint.pth",
    ):
        last_epoch = self.load_checkpoint(path)
        self.fit(epochs, last_epoch + 1, save_frequency, save_path)

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
