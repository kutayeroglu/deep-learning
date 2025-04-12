import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Compute the cross entropy loss.

    Args:
        y_pred (numpy.ndarray): Predicted values, shape (batch_size, num_classes).
            These should be raw logits (pre-softmax).
        y_true (numpy.ndarray): True labels, shape (batch_size,).
            These should be integer class indices.

    Returns:
        tuple: (loss, gradient)
            - loss (float): Average cross entropy loss across the batch
            - gradient (numpy.ndarray): Gradient of the loss with respect to y_pred,
              shape (batch_size, num_classes)
    """
    batch_size = y_pred.shape[0]

    # Apply softmax to get probabilities
    # Subtract max for numerical stability (prevent overflow)
    shifted_logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute cross entropy loss
    # Select the probability of the correct class for each example
    correct_logprobs = -np.log(probs[range(batch_size), y_true])
    loss = np.sum(correct_logprobs) / batch_size

    # Compute gradient
    # Start with the softmax probabilities
    dscores = probs.copy()
    # Subtract 1 from the probability of the correct class
    dscores[range(batch_size), y_true] -= 1
    # Normalize by batch size
    dscores /= batch_size

    return loss, dscores
