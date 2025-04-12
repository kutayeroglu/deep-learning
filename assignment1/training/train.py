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
