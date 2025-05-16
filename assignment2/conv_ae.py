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
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import custom modules
from data.data_loader import load_quickdraw_data
from models.conv_autoencoder import ConvAutoencoder
from training.train import Trainer
from utils.visualization import plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autoencoders for QuickDraw Dataset Embeddings",
    )

    # Default paths
    default_data_dir = os.path.join(SCRIPT_DIR, "data", "quickdraw_subset_np")
    default_save_dir = os.path.join(SCRIPT_DIR, "results")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)

    # Other parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    parser.add_argument("--no_cuda", action="store_true")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Setup device and seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    )

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"conv_autoencoder_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = load_quickdraw_data(
        args.data_dir, args.batch_size, args.val_split, args.seed
    )

    # Create model
    print("Creating model...")
    model = ConvAutoencoder()
    print(model)

    # Set up trainer and optimizer
    trainer = Trainer(model, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training info
    print("\nTraining configuration:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}\n")

    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )

    # Save results
    fig, _ = plot_training_history(history, title_override="CNN-AE MSE Loss")
    fig.savefig(os.path.join(save_dir, "training_history.png"))
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    print("\nTraining completed and results saved!")


if __name__ == "__main__":
    main()
