import numpy as np


class Parameter:
    """Wrapper class for parameters to store the parameter value and its gradient."""

    def __init__(self, value):
        self.value = value
        self.grad = None


class SGDMomentum:
    """Stochastic Gradient Descent with Momentum optimizer."""

    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        """
        Initialize the optimizer.

        Args:
            parameters: List of Parameter objects to optimize
            learning_rate: Learning rate for gradient descent
            momentum: Momentum factor (0 = no momentum)
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [np.zeros_like(param.value) for param in parameters]

    def step(self):
        """Perform one optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Update velocity with momentum
                self.velocity[i] = (
                    self.momentum * self.velocity[i] - self.learning_rate * param.grad
                )

                # Update parameter
                param.value += self.velocity[i]

    def zero_grad(self):
        """Reset gradients to zero."""
        for param in self.parameters:
            param.grad = None
