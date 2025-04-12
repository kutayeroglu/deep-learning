import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin


    def forward(self, image_embs, all_word_embs, labels):
        """
        Compute the contrastive loss for a batch of image embeddings and all word embeddings.
        
        For each image:
        - Positive pair loss: squared Euclidean distance between the image embedding and the correct word embedding.
        - Negative pairs loss: squared hinge loss penalizing negative pairs (i.e. for wrong categories)
        when the distance is below a certain margin.
        
        Args:
            image_embs (Tensor): Tensor of shape (batch_size, embedding_dim) containing the normalized image embeddings.
            all_word_embs (Tensor): Tensor of shape (num_categories, embedding_dim) containing normalized word embeddings.
            labels (Tensor): Tensor of shape (batch_size,) containing the correct category indices.
            margin (float): Margin hyperparameter for the hinge loss on negative pairs.
        
        Returns:
            loss (Tensor): The computed contrastive loss.
        """
        # Calculate distances between each image embedding and every word embedding.
        # (Using p=2 gives us the Euclidean distance.)
        dists = torch.cdist(image_embs, all_word_embs, p=2)  # shape: (batch_size, num_categories)
        
        # For each image, pick the distance corresponding to the correct category label.
        batch_indices = torch.arange(image_embs.size(0), device=image_embs.device)
        pos_dists = dists[batch_indices, labels]  # shape: (batch_size,)
        
        # Positive loss: encourage small distance for the correct pair.
        loss_pos = pos_dists ** 2
        
        # Negative loss:
        # Create a mask where the correct class is False and all negatives are True.
        mask = torch.ones_like(dists, dtype=torch.bool)
        mask[batch_indices, labels] = False
        # Get the distances for all negative pairs, and reshape to (batch_size, num_negative_categories)
        neg_dists = dists[mask].view(image_embs.size(0), -1)
        # Hinge on negative pairs: penalize distances that fall short of the margin.
        loss_neg = F.relu(self.margin - neg_dists) ** 2
        
        # Combine losses.
        # Here we average the positive and negative components.
        loss = (loss_pos.mean() + loss_neg.mean()) / 2.0
        return loss

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
