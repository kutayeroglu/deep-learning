import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE


# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import custom modules
from data.data_loader import load_quickdraw_data
from models.conv_autoencoder import ConvAutoencoder
from training.train import Trainer
from utils.visualization import (
    plot_training_history,
    plot_2d_embeddings,
)
from utils.helpers import extract_embeddings


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
    # Train the model
    print("Training model...")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Patience: {args.patience}")
    print(f"Epochs: {args.epochs}")

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

    # Plot and save training history
    fig, _ = plot_training_history(history, title_override="CNN-AE MSE Loss")
    history_path = os.path.join(save_dir, "training_history.png")
    fig.savefig(history_path)
    print(f"Training history saved to {history_path}")

    # Save model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, test_loader, device)

    # Plot t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plot_2d_embeddings(
        embeddings_2d=tsne_results,
        labels=labels,
        title_override="CNN Autoencoder",
        save_path=os.path.join(save_dir, "cnn_tsne.png"),
    )


if __name__ == "__main__":
    main()
