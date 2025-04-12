import numpy as np

from models.layers import Sequential, Linear, ReLU


class MLP:
    """
    NumPy implementation of a Multi-Layer Perceptron for image classification.
    Input: 784-dimensional vector (flattened 28x28 images)
    Output: 5-dimensional vector (class probabilities)
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
            output_dim (int): Number of output classes (default: 5)
        """
        hidden_layers = []
        prev_dim = input_dim

        # Add hidden layers with ReLU activation
        for dim in hidden_dims:
            hidden_layers.append(Linear(prev_dim, dim))
            hidden_layers.append(ReLU())
            prev_dim = dim

        # Output layer (no activation as we'll apply softmax in predict method, separating forward and probability logic)
        self.output_layer = Linear(prev_dim, output_dim)

        # Combine all hidden layers into a sequential module
        self.hidden_layers = Sequential(*hidden_layers)

    def forward(self, x):
        raise NotImplementedError("Forward pass logic not implemented yet.")

    def predict(self, x):
        raise NotImplementedError("Prediction logic not implemented yet.")
