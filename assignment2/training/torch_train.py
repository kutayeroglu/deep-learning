import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)



class Trainer:
    """
    Trainer class for PyTorch models.
    """

    def __init__(self, model, device=None):
        """
        Initialize the trainer with a model.

        Args:
            model: PyTorch model
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epochs_trained = 0

    def train(
        self,
        train_loader,
        val_loader,
        criterion=None,
        optimizer=None,
        num_epochs=10,
        momentum=0.9,
        lr=0.01,
        patience=5,
        verbose=True,
        margin=1.0,
    ):
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: SGD with momentum)
            num_epochs: Number of epochs to train
            momentum: Momentum value for SGD
            lr: Learning rate
            patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            dict: Dictionary containing training history
        """
        # Set up criterion and optimizer if not provided
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Variables for early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        start_time = time.time()

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_true = []

            train_iter = (
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
                if verbose
                else train_loader
            )

            for inputs, labels in train_iter:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Forward pass: get image embeddings
                image_embs = self.model.image_mlp(inputs)
                image_embs = F.normalize(image_embs, p=2, dim=1)
                
                # Compute the word embeddings for all categories
                all_word_embs = self.model.word_mlp(self.model.category_glove)
                all_word_embs = F.normalize(all_word_embs, p=2, dim=1)

                # loss = criterion(outputs, labels)
                loss = criterion(image_embs, all_word_embs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track loss and predictions
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(labels.cpu().numpy())

            # Calculate average loss and accuracy
            train_loss = train_loss / len(train_loader.dataset)
            train_accuracy = accuracy_score(train_true, train_preds)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader, criterion)

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.epochs_trained += epoch + 1
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        # Return training history
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_accuracy": self.train_accuracies,
            "val_accuracy": self.val_accuracies,
            "training_time": training_time,
        }

        return history

    def evaluate(self, data_loader, criterion=None, margin=1.0):
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for evaluation
            criterion: Loss function

        Returns:
            tuple: (loss, accuracy)
        """
        # Use cross entropy loss if not specified
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()  # Set to evaluation mode

        eval_loss = 0.0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                # loss = criterion(outputs, labels)

                # Forward pass: get image embeddings
                image_embs = self.model.image_mlp(inputs)
                image_embs = F.normalize(image_embs, p=2, dim=1)
                
                # Compute the word embeddings for all categories
                all_word_embs = self.model.word_mlp(self.model.category_glove)
                all_word_embs = F.normalize(all_word_embs, p=2, dim=1)

                # loss = criterion(outputs, labels)
                loss = criterion(image_embs, all_word_embs, labels)


                # Track loss and predictions
                eval_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())

        # Calculate average loss and accuracy
        eval_loss = eval_loss / len(data_loader.dataset)
        accuracy = accuracy_score(all_true, all_preds)

        return eval_loss, accuracy

    def predict(self, data_loader):
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for prediction

        Returns:
            tuple: (predictions, true_labels, probabilities)
        """
        self.model.eval()  # Set to evaluation mode

        all_preds = []
        all_true = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)

                # Get predictions
                _, preds = torch.max(outputs, 1)

                # Save results
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        return np.array(all_preds), np.array(all_true), np.array(all_probs)

    def evaluate_metrics(self, data_loader):
        """
        Evaluate the model with multiple metrics.

        Args:
            data_loader: DataLoader for evaluation

        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Generate predictions
        preds, true_labels, probs = self.predict(data_loader)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, preds)

        # For multi-class, we need to use one-vs-rest approach for some metrics
        n_classes = probs.shape[1]

        # Convert to one-hot encoding for ROC
        true_one_hot = np.zeros((len(true_labels), n_classes))
        true_one_hot[np.arange(len(true_labels)), true_labels] = 1

        # Calculate precision and recall (weighted average for multi-class)
        precision = precision_score(
            true_labels, preds, average="weighted", zero_division=0
        )
        recall = recall_score(true_labels, preds, average="weighted", zero_division=0)

        # Calculate ROC AUC
        roc_auc = roc_auc_score(true_one_hot, probs, multi_class="ovr")

        # Calculate Precision-Recall AUC for each class
        pr_auc_scores = []
        for i in range(n_classes):
            precision_curve, recall_curve, _ = precision_recall_curve(
                true_one_hot[:, i], probs[:, i]
            )
            pr_auc = auc(recall_curve, precision_curve)
            pr_auc_scores.append(pr_auc)

        # Average PR AUC across all classes
        pr_auc = np.mean(pr_auc_scores)

        # Compile results
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

        return metrics

    def plot_training_history(self, figsize=(12, 5)):
        """
        Plot the training history.

        Args:
            figsize: Figure size

        Returns:
            tuple: (figure, axes)
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot losses
        axes[0].plot(self.train_losses, label="Training Loss")
        axes[0].plot(self.val_losses, label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Plot accuracies
        axes[1].plot(self.train_accuracies, label="Training Accuracy")
        axes[1].plot(self.val_accuracies, label="Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        return fig, axes
