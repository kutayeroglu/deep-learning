import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt

from utils.loss_functions import cross_entropy_loss
from training.optimizers import SGDMomentum


class Trainer:
    """
    Trainer class for no-PyTorch MLP
    """

    def __init__(self, model):
        """
        Initialize the trainer with a model.

        Args:
            model: Neural network model
        """
        self.model = model

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
        criterion,
        optimizer=None,
        num_epochs=10,
        momentum=0.9,
        lr=0.01,
        patience=5,
        verbose=True,
    ):
        """
        Train the model.
        """
        if criterion is None:
            criterion = cross_entropy_loss

        if optimizer is None:
            optimizer = SGDMomentum(
                parameters=self.model.parameters(),
                learning_rate=lr,
                momentum=momentum,
            )

        # Variables for early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        start_time = time.time()

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                batch_size = inputs.shape[0]
                train_total += batch_size

                # Forward pass
                outputs = self.model.forward(inputs)

                # Compute loss and gradient (d_L / d_logits)
                loss, grad = criterion(outputs, labels)

                # Track loss & accuracy
                train_loss += loss * batch_size
                preds = np.argmax(outputs, axis=1)
                train_correct += np.sum(preds == labels)

                # ---------------------------
                # Backpropagation - Output Layer
                # ---------------------------
                hidden_output = self.model.hidden_layers.forward(inputs)
                self.model.output_layer.weights.grad = np.dot(hidden_output.T, grad)
                self.model.output_layer.bias.grad = np.sum(grad, axis=0)

                # Gradient w.r.t output of hidden layers
                next_grad = np.dot(grad, self.model.output_layer.weights.value.T)

                # ---------------------------
                # Backpropagation - Hidden Layers (Revised Loop)
                # ---------------------------
                for i in range(len(self.model.hidden_layers.layers) - 1, -1, -2):
                    if (
                        i >= 1
                    ):  # Make sure there is a preceding linear layer for the ReLU layer at index i.
                        linear_layer = self.model.hidden_layers.layers[i - 1]
                        relu_layer = self.model.hidden_layers.layers[i]

                        # Determine the input for the current linear layer.
                        if i > 1:
                            layer_input = self.model.hidden_layers.forward_through(
                                inputs, 0, i - 1
                            )
                        else:
                            layer_input = inputs

                        # TODO: it's more efficient to cache this
                        relu_input = linear_layer.forward(layer_input)

                        # Compute the derivative of ReLU (gradient w.r.t Z)
                        relu_mask = (relu_input > 0).astype(float)
                        # Apply the chain rule
                        dZ = next_grad * relu_mask

                        # Compute gradients for weights and biases
                        linear_layer.weights.grad = np.dot(layer_input.T, dZ)
                        linear_layer.bias.grad = np.sum(dZ, axis=0)

                        # Propagate the gradient to the previous layer
                        next_grad = np.dot(dZ, linear_layer.weights.value.T)

                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

            # Calculate average loss and accuracy
            train_loss /= train_total
            train_accuracy = train_correct / train_total

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
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.epochs_trained += epoch + 1
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        # Return training history
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_acc": self.train_accuracies,
            "val_acc": self.val_accuracies,
            "training_time": training_time,
        }

        return history

    def evaluate(self, data_loader, loss_fn):
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for evaluation
            loss_fn: Loss function

        Returns:
            tuple: (loss, accuracy)
        """
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0

        for inputs, labels in data_loader:
            batch_size = inputs.shape[0]
            eval_total += batch_size

            # Forward pass
            outputs = self.model.forward(inputs)

            # Compute loss
            loss, _ = loss_fn(outputs, labels)

            # Track loss
            eval_loss += loss * batch_size

            # Track accuracy
            preds = np.argmax(outputs, axis=1)
            eval_correct += np.sum(preds == labels)

        # Calculate average loss and accuracy
        eval_loss /= eval_total
        accuracy = eval_correct / eval_total

        return eval_loss, accuracy

    def predict(self, data_loader):
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for prediction

        Returns:
            tuple: (predictions, true_labels, probabilities)
        """
        all_preds = []
        all_true = []
        all_probs = []

        for inputs, labels in data_loader:
            # Forward pass
            outputs = self.model.forward(inputs)

            # Apply softmax to get probabilities
            # Subtract max for numerical stability
            shifted_logits = outputs - np.max(outputs, axis=1, keepdims=True)
            exp_scores = np.exp(shifted_logits)
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Get predictions
            preds = np.argmax(probabilities, axis=1)

            # Save results
            all_preds.extend(preds)
            all_true.extend(labels)
            all_probs.append(probabilities)

        return np.array(all_preds), np.array(all_true), np.concatenate(all_probs)

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
