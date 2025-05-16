import os
import sys
import argparse
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to sys.path to enable importing modules
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Add the assignment2 directory to sys.path
ASSIGNMENT_DIR = SCRIPT_DIR
if not ASSIGNMENT_DIR.endswith("assignment2"):
    ASSIGNMENT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "assignment2")
if ASSIGNMENT_DIR not in sys.path:
    sys.path.append(ASSIGNMENT_DIR)

# Import custom modules
from data.data_loader import load_quickdraw_data
from models.gru_autoencoder import GRUAutoencoder

from training.train import Trainer
from utils.visualization import (
    plot_training_history,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autoencoders for QuickDraw Dataset Embeddings",
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

    # Architecture parameters
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden size for all GRU layers (uniform across layers)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of stacked GRU layers",
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
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA training")

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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Create save directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"autoencoder_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = load_quickdraw_data(
        args.data_dir,
        args.batch_size,
        args.val_split,
        args.seed,
    )
    print(f"Number of classes: {num_classes}")

    # Create model
    print("Creating model...")
    model = GRUAutoencoder(
        input_size=28,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    print(model)

    # Set up trainer
    trainer = Trainer(
        model=model,
        device=device,
        save_dir="quickdraw_autoencoder",
    )

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    # Train the model
    print("Training model...")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Testing on {len(test_loader.dataset)} samples")
    print(f"Using batch size: {args.batch_size}")
    print(f"Using learning rate: {args.lr}")
    print(f"Using momentum: {args.momentum}")
    print(f"Using patience: {args.patience}")
    print(f"Using epochs: {args.epochs}")
    print(f"Using hidden dimension: {args.hidden_dim}")
    print(f"Using number of layers: {args.num_layers}")
    print(f"Using seed: {args.seed}")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )

    # Plot and save training history
    fig, _ = plot_training_history(history, title_override="MSE Loss over Epochs")
    history_path = os.path.join(save_dir, "training_history.png")
    fig.savefig(history_path)
    print(f"Training history saved to {history_path}")

    # Save model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Done!")


if __name__ == "__main__":
    """
    To train a 2-layer GRU with hidden size 256:

        bash
        python train.py --hidden_dim 256 --num_layers 2
    """

    main()
