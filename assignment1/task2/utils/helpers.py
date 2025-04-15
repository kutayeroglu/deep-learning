import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# Function to load GloVe vectors
def load_glove_vectors(category_names, glove_path="glove.6B.50d.txt"):
    """
    Load GloVe vectors for the given category names.
    
    Args:
        category_names: List of category names
        glove_path: Path to the GloVe vectors file
    
    Returns:
        Tensor of shape (len(category_names), glove_dim)
    """
    # Check if the GloVe file exists
    if not os.path.exists(glove_path):
        raise FileNotFoundError(f"The directory '{glove_path}' does not exist.")

    
    # If the file exists, load the embeddings
    word_to_vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]])
            word_to_vec[word] = vector
    
    # Get embeddings for each category
    embeddings = []
    for category in category_names:
        if category in word_to_vec:
            embeddings.append(word_to_vec[category])
        else:
            # the word is not in GloVe
            raise ValueError(f"'{category}' not found in GloVe embeddings.")

    
    return torch.stack(embeddings)
