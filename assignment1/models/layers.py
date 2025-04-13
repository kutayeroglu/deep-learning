import numpy as np
from training.optimizers import Parameter


class Layer:
    """Base class for layers in the network"""

    def forward(self, inputs):
        raise NotImplementedError


class Linear(Layer):
    """Linear (fully connected) layer"""

    def __init__(self, input_dim, output_dim):
        """
        Glorot initialization performs poorly for ReLu activation.
        Source: https://arxiv.org/abs/1704.08863

        He (Kaiming) initialization was proposed for networks with ReLU activation.
        Source: https://arxiv.org/abs/1502.01852
        """
        # Initialize weights with He initialization
        weights_array = np.random.randn(input_dim, output_dim) * np.sqrt(
            2.0 / input_dim
        )
        bias_array = np.zeros(output_dim)

        # Wrap arrays in Parameter objects
        self.weights = Parameter(weights_array)
        self.bias = Parameter(bias_array)

    def forward(self, inputs):
        return np.dot(inputs, self.weights.value) + self.bias.value


class ReLU(Layer):
    """ReLU activation function"""

    def forward(self, inputs):
        return np.maximum(0, inputs)


class Sequential:
    """Container that runs layers sequentially"""

    def __init__(self, *layers):
        self.layers = list(layers) if layers else []

    def forward(self, inputs):
        """Forward pass through all layers"""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def forward_through(self, inputs, start_idx=0, end_idx=None):
        """Forward pass through a specific subset of layers

        Args:
            inputs: Input tensor
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (exclusive)

        Returns:
            Output tensor after passing through the specified layers
        """
        if end_idx is None:
            end_idx = len(self.layers)

        for i in range(start_idx, end_idx):
            inputs = self.layers[i].forward(inputs)
        return inputs
