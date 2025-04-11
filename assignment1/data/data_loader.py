import numpy as np
import os
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Dataset class for handling image data and labels.
    """

    def __init__(self, images, labels, transform=None):
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
        """Get a single item from the dataset."""
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Flatten the image if it's not already flattened
        if image.ndim > 1:
            image = image.reshape(-1)

        return image, label


class DataLoader:
    """
    DataLoader class for batching, shuffling, and iterating through the dataset.
    """

    def __init__(self, dataset, batch_size=32, shuffle=False, seed=None):
        """
        Initialize the data loader.

        Args:
            dataset (Dataset): Dataset to load data from
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the dataset before each epoch
            seed (int, optional): Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        # Generate initial indices
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            self.rng.shuffle(self.indices)

        # Calculate number of batches
        self.num_batches = int(np.ceil(len(dataset) / batch_size))
        self.current_batch = 0

    def __len__(self):
        """Return the number of batches."""
        return self.num_batches


def load_quickdraw_data(data_dir, batch_size=64, validation_split=0.1, seed=42):
    """
    Load the QuickDraw dataset and create DataLoaders.

    Args:
        data_dir (str): Directory containing the dataset files
        batch_size (int): Batch size for DataLoaders
        validation_split (float): Fraction of training data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load the data
    train_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    test_images = np.load(os.path.join(data_dir, "test_images.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Split training data into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images,
        train_labels,
        test_size=validation_split,
        random_state=seed,
        stratify=train_labels,
    )

    # Create datasets
    train_dataset = Dataset(train_images, train_labels)
    val_dataset = Dataset(val_images, val_labels)
    test_dataset = Dataset(test_images, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, seed=seed
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, seed=seed
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
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break  # Just look at the first batch
