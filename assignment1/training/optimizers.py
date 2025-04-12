import numpy as np


class SGDMomentum:
    """
    Stochastic Gradient Descent optimizer with momentum.

    The update rule is:
    velocity = momentum * velocity - learning_rate * gradient
    parameter = parameter + vvelocity
    """

    def __init__(self, parameters=None, learning_rate=0.01, momentum=0.9):
        """
        Initialize the optimizer.

        Args:
            parameters: List of parameter arrays to optimize (optional)
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient (0 <= momentum < 1)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        self.parameters = None

        # Initialize with parameters if provided
        if parameters is not None:
            self.initialize(parameters)

    def initialize(self, parameters):
        """
        Initialize optimizer state with model parameters.

        Args:
            parameters: List of parameter arrays to optimize
        """
        self.parameters = parameters
        # Initialize velocities to zeros with same shapes as parameters
        self.velocities = {id(param): np.zeros_like(param) for param in parameters}

    def step(self):
        """
        Perform one optimization step.
        Updates parameters based on their gradients.
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Reset all parameter gradients to zero.
        """
        raise NotImplementedError
