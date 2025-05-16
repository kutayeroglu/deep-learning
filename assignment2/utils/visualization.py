import matplotlib.pyplot as plt


def plot_training_history(history, title_override=None):
    """Plot training history."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot loss
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Validation")
    ax.set_title("Loss" if title_override is None else title_override)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.tight_layout()
    return fig, ax
