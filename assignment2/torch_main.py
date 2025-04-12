import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from training.torch_train import Trainer
from utils.helpers import ContrastiveLoss, load_glove_vectors
from models.torch_mlp import JointModel, TorchMLP

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add parent directory to path
sys.path.append(os.path.dirname(SCRIPT_DIR))


# Function to plot sample predictions
def plot_sample_predictions(test_loader, model, device, save_dir, num_samples=5):
    """Plot a sample of test images with predictions."""
    # Get a batch of images
    images, labels = next(iter(test_loader))

    # Select a subset of images
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Get predictions
    model.eval()
    with torch.no_grad():
        images_flat = images.view(images.size(0), -1).to(device)
        similarities = model(images_flat)
        probs = torch.softmax(similarities, dim=1)
        preds = similarities.argmax(dim=1)

    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()

    # Plot images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        # Reshape the image
        img = images[i].reshape(28, 28)

        # Plot the image
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"True: {labels[i]}\nPred: {preds[i]}\nProb: {probs[i][preds[i]]:.2f}"
        )
        axes[i].axis("off")

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Sample predictions saved to {plot_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Joint Model for QuickDraw Classification")

    # Default paths relative to script location
    default_data_dir = os.path.join(SCRIPT_DIR, "data", "quickdraw_subset_np")
    default_save_dir = os.path.join(SCRIPT_DIR, "results")
    default_glove_dir = os.path.join(SCRIPT_DIR, "glove.6B")

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
        "--hidden_dims_imageMLP",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Dimensions of hidden layers of Image MLP",
    )
    parser.add_argument(
        "--hidden_dims_wordMLP",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Dimensions of hidden layers of Word MLP",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Dimension of the joint embedding space",
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        default=os.path.join(default_glove_dir, "glove.6B.50d.txt"),
        help="Path to GloVe vectors file",
    )

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
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
    save_dir = os.path.join(args.save_dir, f"joint_model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    from data.torch_loader import load_quickdraw_data
    train_loader, val_loader, test_loader, num_classes = load_quickdraw_data(
        args.data_dir, args.batch_size, args.val_split, args.seed
    )
    print(f"Number of classes: {num_classes}")

    # Get category names (would typically be loaded from data)
    # For this example, we'll use placeholder names
    category_names = ["rabbit", "yoga", "hand", "snowman", "motorbike"]
    if len(category_names) != num_classes:
        print(f"Warning: Placeholder category names used. Expected {num_classes} classes, got {len(category_names)}")
        # Create generic category names if needed
        category_names = [f"category_{i}" for i in range(num_classes)]

    # Load GloVe vectors
    print("Loading GloVe vectors...")
    category_glove = load_glove_vectors(category_names, args.glove_path)
    print(f"GloVe vectors shape: {category_glove.shape}")

    # Create models
    print("Creating model...")
    image_mlp = TorchMLP(
        input_dim=784,  # 28x28 flattened images
        hidden_dims=args.hidden_dims_imageMLP,
        output_dim=args.embedding_dim,
    ).to(device)
    
    word_mlp = TorchMLP(
        input_dim=category_glove.shape[1],  # GloVe dimension
        hidden_dims=args.hidden_dims_wordMLP,
        output_dim=args.embedding_dim,
    ).to(device)
    
    model = JointModel(
        image_mlp=image_mlp, 
        word_mlp=word_mlp, 
        category_glove=category_glove.to(device)
    ).to(device)
    
    print("Model created:")
    print("Image MLP:", image_mlp)
    print("Word MLP:", word_mlp)

    # Set up trainer and optimizer
    # trainer = JointTrainer(model, device)
    trainer = Trainer(model, device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train the model
    print("Training model...")
    criterion = ContrastiveLoss()
    history = trainer.train(
        train_loader,
        val_loader,
        optimizer=optimizer,
        criterion=criterion,
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
    fig, _ = trainer.plot_training_history()
    history_path = os.path.join(save_dir, "training_history.png")
    fig.savefig(history_path)
    print(f"Training history saved to {history_path}")

    # Save model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Test Metrics:\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

        f.write("\nModel Architecture:\n")
        f.write(f"Image MLP: {image_mlp}\n")
        f.write(f"Word MLP: {word_mlp}\n")

        f.write("\nTraining Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Metrics saved to {metrics_path}")

    # Plot a sample of test images with predictions
    print("Generating sample predictions...")
    plot_sample_predictions(test_loader, model, device, save_dir)

    print("Done!")

if __name__ == "__main__":
    main()