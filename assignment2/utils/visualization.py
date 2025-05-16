import os

import matplotlib.pyplot as plt


def plot_roc_pr_curves(metrics, save_dir):
    """
    Plot ROC and Precision-Recall curves for all classes.

    Args:
        metrics: Dictionary containing metrics and curve data
        save_dir: Directory to save the plots
    """
    # Extract curve data
    roc_curves = metrics["roc_curves"]
    pr_curves = metrics["pr_curves"]
    roc_aucs = metrics["roc_aucs"]
    pr_aucs = metrics["pr_aucs"]

    # Create figure for ROC curves
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    for idx, (fpr, tpr, class_idx) in enumerate(roc_curves):
        plt.plot(fpr, tpr, lw=2, label=f"Class {class_idx} (AUC = {roc_aucs[idx]:.3f})")

    # Plot random chance line
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random (AUC = 0.500)")

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"Receiver Operating Characteristic (ROC) Curves\nAverage AUC = {metrics['roc_auc']:.3f}"
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save the plot
    roc_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curves saved to {roc_path}")

    # Create figure for Precision-Recall curves
    plt.figure(figsize=(10, 8))

    # Plot PR curve for each class
    for idx, (precision_curve, recall_curve, class_idx) in enumerate(pr_curves):
        plt.plot(
            recall_curve,
            precision_curve,
            lw=2,
            label=f"Class {class_idx} (AUC = {pr_aucs[idx]:.3f})",
        )

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves\nAverage AUC = {metrics['pr_auc']:.3f}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)

    # Save the plot
    pr_path = os.path.join(save_dir, "pr_curves.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Precision-Recall curves saved to {pr_path}")


def plot_training_history(history, title_override=None):
    """Plot training history."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot loss
    ax[0].plot(history["train_loss"], label="Train")
    ax[0].plot(history["val_loss"], label="Validation")
    ax[0].set_title("Loss" if title_override is None else title_override)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    plt.tight_layout()
    return fig, ax
