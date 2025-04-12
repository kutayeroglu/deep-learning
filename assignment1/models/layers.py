import numpy as np


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
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias


class ReLU(Layer):
    """ReLU activation function"""

    def forward(self, inputs):
        return np.maximum(0, inputs)


class Sequential:
    """Container that runs layers sequentially"""

    def __init__(self, *layers):
        self.layers = list(layers) if layers else []

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
