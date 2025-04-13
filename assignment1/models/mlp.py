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

        # Output layer: produces raw logits; softmax will be applied in the loss function
        self.output_layer = Linear(prev_dim, output_dim)

        # Combine all hidden layers into a sequential module
        self.hidden_layers = Sequential(*hidden_layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (numpy.ndarray): Input data, shape (batch_size, input_dim) or (batch_size, height, width)

        Returns:
            numpy.ndarray: Output logits, shape (batch_size, output_dim)
        """
        # Ensure the input is 2D (batch_size, input_dim)
        if x.ndim > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        # Pass through hidden layers
        hidden_output = self.hidden_layers.forward(x)

        # Pass through output layer
        output = self.output_layer.forward(hidden_output)

        return output

    def predict(self, x):
        """
        Predict class labels for input data.

        Args:
            x (numpy.ndarray): Input data, shape (batch_size, input_dim) or (batch_size, height, width)

        Returns:
            numpy.ndarray: Predicted class indices, shape (batch_size,)
        """
        # Forward pass to get logits
        logits = self.forward(x)

        # Get the class with highest logit (argmax)
        predictions = np.argmax(logits, axis=1)

        return predictions

    def parameters(self):
        """
        Get all trainable parameters of the model.

        Returns:
            list: List of parameter objects
        """
        params = []

        # Add parameters from hidden layers that have weights and biases
        for layer in self.hidden_layers.layers:
            if hasattr(layer, "weights"):
                params.append(layer.weights)
                params.append(layer.bias)

        # Add parameters from the output layer
        params.append(self.output_layer.weights)
        params.append(self.output_layer.bias)

        return params

    def save(self, path):
        """
        Save model parameters to a file.

        Args:
            path (str): Path to save the model
        """
        # Create a dictionary to store all weights and biases
        params_dict = {}

        # Save hidden layer parameters
        for i, layer in enumerate(self.hidden_layers.layers):
            if hasattr(layer, "weights"):
                params_dict[f"hidden_layer_{i}_weights"] = layer.weights.value
                params_dict[f"hidden_layer_{i}_bias"] = layer.bias.value

        # Save output layer parameters
        params_dict["output_layer_weights"] = self.output_layer.weights.value
        params_dict["output_layer_bias"] = self.output_layer.bias.value

        # Save to file
        np.savez(path, **params_dict)

    # def load(self, path):
    #     """
    #     Load model parameters from a file.

    #     Args:
    #         path (str): Path to load the model from
    #     """
    #     params = np.load(path)

    #     # Load hidden layer parameters
    #     for i, layer in enumerate(self.hidden_layers.layers):
    #         if hasattr(layer, "weights"):
    #             layer.weights.value = params[f"hidden_layer_{i}_weights"]
    #             layer.bias.value = params[f"hidden_layer_{i}_bias"]

    #     # Load output layer parameters
    #     self.output_layer.weights.value = params["output_layer_weights"]
    #     self.output_layer.bias.value = params["output_layer_bias"]
