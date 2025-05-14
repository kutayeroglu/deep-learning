import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class QuickDrawDataset(Dataset):
    """
    PyTorch Dataset for the QuickDraw subset.
    """

    def __init__(
        self,
        images,
        labels,
        transform=None,
    ):
        """
        Initialize the dataset.

        Args:
            images (numpy.ndarray): Image data with shape (n_samples, height, width)
            labels (numpy.ndarray): Labels with shape (n_samples,)
            transform (callable, optional): Optional transform to apply to the data
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Convert to tensor if not already
        image_tensor = (
            torch.from_numpy(image).float()
            if not isinstance(image, torch.Tensor)
            else image
        )
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Flatten the image if it's not already flattened
        if image_tensor.dim() > 1:
            image_tensor = image_tensor.view(-1)

        return image_tensor, label_tensor


def load_quickdraw_data(
    data_dir,
    batch_size=64,
    validation_split=0.1,
    seed=42,
):
    """
    Load the QuickDraw dataset and create PyTorch DataLoaders.

    Args:
        data_dir (str): Directory containing the dataset files
        batch_size (int): Batch size for DataLoaders
        validation_split (float): Fraction of training data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load the data
    train_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    test_images = np.load(os.path.join(data_dir, "test_images.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Create the full training dataset
    full_train_dataset = QuickDrawDataset(train_images, train_labels)

    # Split into training and validation
    val_size = int(len(full_train_dataset) * validation_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create the test dataset
    test_dataset = QuickDrawDataset(test_images, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # Determine the number of classes
    num_classes = len(np.unique(train_labels))

    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    # Example usage
    data_dir = "quickdraw_subset_np"
    train_loader, val_loader, test_loader, num_classes = load_quickdraw_data(data_dir)

    # Print some information
    print(f"Number of classes: {num_classes}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Inspect one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
