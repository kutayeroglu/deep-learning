import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


class Trainer:
    def __init__(self, model, device, save_dir="checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def train(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=10,
        patience=3,
        verbose=True,
    ):
        best_loss = np.inf
        epochs_no_improve = 0
        best_model_wts = None

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=not verbose,
            ):
                inputs, _ = batch  # We don't need labels for autoencoder
                inputs = inputs.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)  # Compare reconstruction to original

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # Validation phase
            val_loss = self.evaluate(val_loader, criterion)

            # Calculate epoch metrics
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = self.model.state_dict().copy()
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping after {epoch + 1} epochs!")
                    break

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.history

    def evaluate(self, loader, criterion):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                inputs, _ = batch
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)

                running_loss += loss.item() * inputs.size(0)

        return running_loss
