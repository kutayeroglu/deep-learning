import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchMLP(nn.Module):
    """
    PyTorch implementation of a Multi-Layer Perceptron for image classification.
    Input: 784-dimensional vector (flattened 28x28 images)
    """

    def __init__(
        self,
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=5,
    ):
        """
        Initialize the MLP with configurable architecture.

        Args:
            input_dim (int): Dimension of input features (default: 784 for 28x28 images)
            hidden_dims (list): List of hidden layer dimensions
        """
        super().__init__()

        # Create the layer structure based on provided dimensions
        layers = []
        prev_dim = input_dim

        # Add hidden layers with ReLU activation
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        # Output layer (no activation as we'll use cross_entropy loss which includes softmax)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Combine all hidden layers into a sequential module
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, output_dim)
        """
        # Ensure input has the right shape
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten if needed

        # Pass through hidden layers
        x = self.hidden_layers(x)

        # Pass through output layer
        logits = self.output_layer(x)

        return logits

    def predict(self, x):
        """
        Make predictions by applying softmax to the model output.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Probability distribution over classes
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


class JointModel(nn.Module):
    def __init__(self, image_mlp, word_mlp, category_glove):
        super().__init__()
        self.image_mlp = image_mlp
        self.word_mlp = word_mlp
        self.category_glove = category_glove  # Shape: (num_categories, glove_dim)

    def forward(self, images, labels=None):
        image_embs = self.image_mlp(images)
        image_embs = F.normalize(image_embs, p=2, dim=1)

        if labels is not None:
            # Training mode
            batch_glove = self.category_glove[labels]  # Get GloVe for batch labels
            word_embs = self.word_mlp(batch_glove)
            word_embs = F.normalize(word_embs, p=2, dim=1)
            return image_embs, word_embs
        else:
            # Inference mode
            all_word_embs = self.word_mlp(self.category_glove)
            all_word_embs = F.normalize(all_word_embs, p=2, dim=1)
            # Calculate similarities
            similarities = image_embs @ all_word_embs.T
            return similarities
