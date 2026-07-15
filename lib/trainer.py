"""Reusable training/evaluation loop for PyTorch classification models.

This module provides `Trainer`, a small helper class that wraps a standard
supervised training loop, a held-out evaluation loop, per-epoch loss/accuracy
tracking, and a convenience plot of that history. It is intended to be
imported across the paper-implementation scripts/notebooks in this repo
(e.g. `lenet-5/lenet_5.py`) rather than copy-pasted into each one.

Scope: this class assumes a classification task where `model(inputs)`
returns raw class scores/logits suitable for `argmax(dim=1)`, and accuracy
is always computed this way. It is not intended for other task types
(e.g. seq2seq, VAE, GAN) — that generalization is explicitly out of scope
for now.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
import matplotlib.pyplot as plt


class Trainer:
    """Trains and evaluates a PyTorch classification model.

    For a configured number of epochs, `fit()` runs one training pass over
    `train_dataloader` followed by one evaluation pass over
    `test_dataloader`, records per-epoch loss/accuracy in `self.history`,
    and (if configured) steps `scheduler` once per epoch. Metrics can be
    visualized afterwards with `plot()`.

    Attributes:
        device (str): Device the model and batches are moved to ("cuda",
            "mps", or "cpu"); auto-detected if not passed explicitly.
        train_dataloader (DataLoader): Yields `(inputs, targets)` batches
            used to update model weights.
        test_dataloader (DataLoader): Yields `(inputs, targets)` batches
            used for held-out evaluation after each epoch.
        model (nn.Module): The model being trained, moved to `device`.
        optimizer (Optimizer): Optimizer used to update `model`'s
            parameters.
        loss_fn (nn.Module): Loss function, called as
            `loss_fn(predictions, targets)`.
        scheduler (lr_scheduler.LRScheduler | None): Optional learning-rate
            scheduler. If set, `step()` is called once per epoch (after
            that epoch's training and evaluation), never per batch.
        log_every (int): Print a training-loss line every `log_every`
            batches within an epoch.
        history (dict[str, list[float]]): Per-epoch metrics recorded by
            `fit()`, with keys "train_loss", "train_accuracy",
            "test_loss", "test_accuracy". Each list has one entry per
            completed epoch, in order.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: str | None = None,
        scheduler: lr_scheduler.LRScheduler | None = None,
        log_every: int = 100,
    ):
        """Initializes the trainer and moves the model to the target device.

        Args:
            train_dataloader: Dataloader providing `(inputs, targets)`
                batches used for training.
            test_dataloader: Dataloader providing `(inputs, targets)`
                batches used for evaluation.
            model: The model to train. Moved to `device` in place.
            loss_fn: Loss function called as `loss_fn(predictions, targets)`.
            optimizer: Optimizer that updates `model`'s parameters.
            device: Target device string ("cuda", "mps", "cpu"). If `None`,
                auto-detects the best available device.
            scheduler: Optional learning-rate scheduler. If provided, its
                `step()` is called once per epoch, after that epoch's
                training and evaluation passes complete (not once per
                batch).
            log_every: Print a training-loss/progress line every
                `log_every` batches. Values `<= 0` disable per-batch
                logging.
        """
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.log_every = log_every

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

    def _train_loop(self) -> tuple[float, float]:
        """Runs one epoch of training over `train_dataloader`.

        Puts the model in training mode and, for each batch, runs a
        forward pass, computes the loss, backpropagates, and updates
        weights via `self.optimizer`. Does not step `self.scheduler`;
        that happens once per epoch in `fit()`.

        Returns:
            A `(train_loss, train_accuracy)` tuple, where `train_loss` is
            the mean loss over all batches and `train_accuracy` is the
            percentage (0-100) of correctly classified training examples.
        """
        train_set_size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        self.model.train()

        train_loss, correct_preds = 0.0, 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            correct_preds += (pred.argmax(1) == y).type(torch.float).sum().item()

            if self.log_every > 0 and batch % self.log_every == 0:
                current = batch * len(X)
                print(
                    f"Training Loss: {loss.item():>7f} [{current:>5d}/{train_set_size:>5d}]"
                )

        train_loss /= num_batches
        train_acc = 100 * correct_preds / train_set_size
        return train_loss, train_acc

    def _test_loop(self) -> tuple[float, float]:
        """Runs one epoch of evaluation over `test_dataloader`.

        Puts the model in eval mode and disables gradient tracking.
        Prints a summary line with the average loss and accuracy for
        the epoch.

        Returns:
            A `(test_loss, test_accuracy)` tuple, where `test_loss` is the
            mean loss over all batches and `test_accuracy` is the
            percentage (0-100) of correctly classified test examples.
        """
        test_set_size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()

        test_loss, correct_preds = 0.0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct_preds += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        test_acc = 100 * correct_preds / test_set_size

        print(f"Test Error: \nAccuracy: {test_acc:>0.1f}% Avg loss: {test_loss:>8f}\n")
        return test_loss, test_acc

    def fit(self, epochs: int) -> None:
        """Trains and evaluates the model for the given number of epochs.

        For each epoch: runs one training pass (`_train_loop`), then one
        evaluation pass (`_test_loop`), appends both results to
        `self.history`, and (if configured) steps `self.scheduler` once.

        Args:
            epochs: Number of epochs to train for.
        """
        print("--------------- Training Loop ---------------\n")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n ---------------------------------------")
            train_loss, train_acc = self._train_loop()
            test_loss, test_acc = self._test_loop()

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_accuracy"].append(test_acc)

            if self.scheduler is not None:
                self.scheduler.step()

    def plot(self) -> None:
        """Plots recorded train/test loss and accuracy curves.

        Requires at least one prior call to `fit()`. Renders two
        side-by-side line charts (loss, accuracy) against epoch number
        using the contents of `self.history`.
        """
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
