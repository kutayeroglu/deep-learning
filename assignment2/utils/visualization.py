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


def plot_2d_embeddings(embeddings_2d, labels, title_override=None, save_path=None):
    """
    Plot 2D embeddings using t-SNE visualization.

    Args:
        embeddings_2d (np.ndarray): 2D array of shape (n_samples, 2)
        labels (np.ndarray): Array of labels for each sample
        title_override (str, optional): Custom title for the plot
        save_path (str, optional): Path to save the figure. If None, shows interactively.
    """
    # Ensure labels are integers for discrete color mapping
    labels = labels.astype(int)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(scatter, label="Class")
    title = title_override if title_override else "Autoencoder Embeddings"
    plt.title(f"t-SNE Visualization: {title}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
