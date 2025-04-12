import os
import sys
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to sys.path to enable importing modules
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Add the assignment1 directory to sys.path
ASSIGNMENT_DIR = SCRIPT_DIR
if not ASSIGNMENT_DIR.endswith("assignment1"):
    ASSIGNMENT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "assignment1")
if ASSIGNMENT_DIR not in sys.path:
    sys.path.append(ASSIGNMENT_DIR)

# Import custom modules
from data.data_loader import load_quickdraw_data
from models.mlp import MLP
from training.train import Trainer
from utils.loss_functions import cross_entropy_loss
from training.optimizers import SGDMomentum


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="From-scratch MLP for QuickDraw Classification"
    )

    # Default paths relative to script location
    default_data_dir = os.path.join(SCRIPT_DIR, "data", "quickdraw_subset_np")
    default_save_dir = os.path.join(SCRIPT_DIR, "results")

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation",
    )

    # Model parameters
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Dimensions of hidden layers",
    )

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )

    # Other parameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=default_save_dir,
        help="Directory to save results",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Print paths for debugging
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Assignment directory: {ASSIGNMENT_DIR}")
    print(f"Data directory: {args.data_dir}")
    print(f"Current working directory: {os.getcwd()}")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)

    # Create save directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"from_scratch_mlp_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = load_quickdraw_data(
        args.data_dir, args.batch_size, args.val_split, args.seed
    )
    print(f"Number of classes: {num_classes}")

    # Create model
    print("Creating model...")
    model = MLP(
        input_dim=784,  # 28x28 images
        hidden_dims=args.hidden_dims,
        output_dim=num_classes,
    )
    print(model)

    # Set up trainer
    trainer = Trainer(model)

    # Set up optimizer
    optimizer = SGDMomentum(
        parameters=model.parameters(), learning_rate=args.lr, momentum=args.momentum
    )

    # Train the model
    history = trainer.train(
        train_loader,
        val_loader,
        loss_fn=cross_entropy_loss,
        optimizer=optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    metrics = trainer.evaluate_metrics(test_loader)

    # Print metrics
    print("\nTest Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Plot and save training history
    fig, ax = plot_training_history(history)
    history_path = os.path.join(save_dir, "training_history.png")
    fig.savefig(history_path)
    print(f"Training history saved to {history_path}")

    # Save model
    model_path = os.path.join(save_dir, "model.npz")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Test Metrics:\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

        f.write("\nModel Architecture:\n")
        f.write(str(model))

        f.write("\nTraining Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Metrics saved to {metrics_path}")

    # Plot a sample of test images with predictions
    print("Generating sample predictions...")
    plot_sample_predictions(test_loader, model, save_dir)

    print("Done!")


def plot_training_history(history):
    """Plot training history."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax[0].plot(history["train_loss"], label="Train")
    ax[0].plot(history["val_loss"], label="Validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(history["train_acc"], label="Train")
    ax[1].plot(history["val_acc"], label="Validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()
    return fig, ax


def plot_sample_predictions(test_loader, model, save_dir, num_samples=5):
    """Plot a sample of test images with predictions."""
    # Get a batch of images
    for images, labels in test_loader:
        break  # Just get the first batch

    # Select a subset of images
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Get predictions
    outputs = model.forward(images)
    probs = model.softmax(outputs)
    preds = np.argmax(probs, axis=1)

    # Plot images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        # Reshape the image
        img = images[i].reshape(28, 28)

        # Plot the image
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"True: {labels[i]}\nPred: {preds[i]}\nProb: {probs[i, preds[i]]:.2f}"
        )
        axes[i].axis("off")

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Sample predictions saved to {plot_path}")


if __name__ == "__main__":
    main()
