import os

import numpy as np


class DataLoader:
    def __init__(self, data_folder, batch_size=32, shuffle=True):
        self.data_dir = os.path.join("data", data_folder)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0

        # Load data
        self.train_images = np.load(os.path.join(self.data_dir, "train_images.npy"))
        self.train_labels = np.load(os.path.join(self.data_dir, "train_labels.npy"))
        self.test_images = np.load(os.path.join(self.data_dir, "test_images.npy"))
        self.test_labels = np.load(os.path.join(self.data_dir, "test_labels.npy"))

        self.num_train_samples = len(self.train_images)
        self.num_test_samples = len(self.test_images)
        self.train_indices = np.arange(self.num_train_samples)
        self.test_indices = np.arange(self.num_test_samples)

        if self.shuffle:
            np.random.shuffle(self.train_indices)
