import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import sys


# Import the model classes
from models.gru_autoencoder import GRUAutoencoder
from models.conv_autoencoder import ConvAutoencoder
from data.data_loader import load_quickdraw_data

def load_model(model_path, model_class, device):
    """Load a trained model from checkpoint"""
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_embeddings_gru(model, data_loader, device):
    """Extract embeddings from the GRU encoder"""
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            
            # Reshape for GRU if needed (B, 28, 28) -> (B, 28, 28)
            if batch_data.dim() == 4:  # If shape is [B, 1, 28, 28]
                batch_data = batch_data.squeeze(1)
            
            # Get embeddings from encoder - GRU returns (output, hidden)
            _, hidden = model.encoder(batch_data)
            
            # Extract the last layer's hidden state for all samples in batch
            # hidden shape: (num_layers, batch_size, hidden_dim)
            embedding = hidden[-1]  # Shape: (batch_size, hidden_dim)
            
            # Convert to numpy and append to lists
            batch_embeddings = embedding.cpu().numpy()
            batch_labels_np = batch_labels.numpy()
            
            all_embeddings.append(batch_embeddings)
            all_labels.append(batch_labels_np)
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # Verify shapes match
    assert embeddings.shape[0] == labels.shape[0], f"Mismatch: embeddings shape {embeddings.shape}, labels shape {labels.shape}"
    
    return embeddings, labels

def get_embeddings_conv(model, data_loader, device):
    """Extract embeddings from the Conv encoder"""
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            
            # Ensure proper shape for Conv (B, 1, 28, 28)
            if batch_data.dim() == 3:  # If shape is [B, 28, 28]
                batch_data = batch_data.unsqueeze(1)
            
            # Get embeddings from encoder
            embedding = model.encoder(batch_data)
            
            # Flatten the spatial dimensions to get a feature vector for each sample
            embedding = embedding.view(embedding.size(0), -1)  # Shape: (batch_size, 128*3*3)
            
            # Convert to numpy and append to lists
            batch_embeddings = embedding.cpu().numpy()
            batch_labels_np = batch_labels.numpy()
            
            all_embeddings.append(batch_embeddings)
            all_labels.append(batch_labels_np)
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # Verify shapes match
    assert embeddings.shape[0] == labels.shape[0], f"Mismatch: embeddings shape {embeddings.shape}, labels shape {labels.shape}"
    
    return embeddings, labels

def apply_tsne(embeddings, perplexity=30, n_iter=1000):
    """Apply t-SNE to reduce dimensionality to 2D"""
    print(f"Applying t-SNE on embeddings of shape {embeddings.shape}")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    return tsne.fit_transform(embeddings)

def plot_tsne(tsne_results, labels, title, save_path):
    """Plot t-SNE results with color-coded labels"""
    print(f"Plotting t-SNE results of shape {tsne_results.shape} with labels of shape {labels.shape}")
    
    # Verify shapes match
    assert tsne_results.shape[0] == labels.shape[0], f"Mismatch: tsne_results shape {tsne_results.shape}, labels shape {labels.shape}"
    
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Create a scatter plot for each class
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            tsne_results[mask, 0], 
            tsne_results[mask, 1],
            label=f'Class {label}',
            alpha=0.6
        )
    
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def analyze_discriminability(tsne_results, labels):
    """Analyze the discriminability of the embeddings"""
    # Calculate the centroid for each class
    unique_labels = np.unique(labels)
    centroids = {}
    
    for label in unique_labels:
        mask = labels == label
        centroids[label] = np.mean(tsne_results[mask], axis=0)
    
    # Calculate average intra-class distance (compactness)
    intra_class_distances = []
    for label in unique_labels:
        mask = labels == label
        points = tsne_results[mask]
        centroid = centroids[label]
        
        # Calculate distances from each point to its centroid
        distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        avg_distance = np.mean(distances)
        intra_class_distances.append(avg_distance)
    
    avg_intra_class_distance = np.mean(intra_class_distances)
    
    # Calculate average inter-class distance (separation)
    inter_class_distances = []
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i < j:  # Avoid duplicate pairs
                distance = np.sqrt(np.sum((centroids[label1] - centroids[label2]) ** 2))
                inter_class_distances.append(distance)
    
    avg_inter_class_distance = np.mean(inter_class_distances)
    
    # Calculate discriminability ratio (higher is better)
    discriminability_ratio = avg_inter_class_distance / avg_intra_class_distance
    
    analysis = {
        'avg_intra_class_distance': avg_intra_class_distance,
        'avg_inter_class_distance': avg_inter_class_distance,
        'discriminability_ratio': discriminability_ratio
    }
    
    return analysis

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model paths
    gru_model_path = 'quickdraw_autoencoder/best_model.pth'
    conv_model_path = 'checkpoints/best_model.pth'

        

    # Load test data
    _, _, test_loader, _ = load_quickdraw_data(
        data_dir='data/quickdraw_subset_np',
        batch_size=64
    )
    
    # Create output directory
    output_dir = 'tsne_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process GRU autoencoder
    print("Processing GRU autoencoder...")
    gru_model = load_model(gru_model_path, GRUAutoencoder, device)
    gru_embeddings, gru_labels = get_embeddings_gru(gru_model, test_loader, device)
    print(f"GRU embeddings shape: {gru_embeddings.shape}, labels shape: {gru_labels.shape}")
    
    gru_tsne = apply_tsne(gru_embeddings)
    gru_plot_path = plot_tsne(gru_tsne, gru_labels, 'GRU Autoencoder Embeddings (t-SNE)', f'{output_dir}/gru_tsne.png')
    gru_analysis = analyze_discriminability(gru_tsne, gru_labels)
    
    # Process Conv autoencoder
    print("Processing Convolutional autoencoder...")
    conv_model = load_model(conv_model_path, ConvAutoencoder, device)
    conv_embeddings, conv_labels = get_embeddings_conv(conv_model, test_loader, device)
    print(f"Conv embeddings shape: {conv_embeddings.shape}, labels shape: {conv_labels.shape}")
    
    conv_tsne = apply_tsne(conv_embeddings)
    conv_plot_path = plot_tsne(conv_tsne, conv_labels, 'Convolutional Autoencoder Embeddings (t-SNE)', f'{output_dir}/conv_tsne.png')
    conv_analysis = analyze_discriminability(conv_tsne, conv_labels)
    
    # Print analysis results
    print("\nGRU Autoencoder Analysis:")
    print(f"Average intra-class distance: {gru_analysis['avg_intra_class_distance']:.4f}")
    print(f"Average inter-class distance: {gru_analysis['avg_inter_class_distance']:.4f}")
    print(f"Discriminability ratio: {gru_analysis['discriminability_ratio']:.4f}")
    
    print("\nConvolutional Autoencoder Analysis:")
    print(f"Average intra-class distance: {conv_analysis['avg_intra_class_distance']:.4f}")
    print(f"Average inter-class distance: {conv_analysis['avg_inter_class_distance']:.4f}")
    print(f"Discriminability ratio: {conv_analysis['discriminability_ratio']:.4f}")
    
    # Compare discriminability
    if gru_analysis['discriminability_ratio'] > conv_analysis['discriminability_ratio']:
        print("\nThe GRU autoencoder produces more discriminable embeddings.")
    else:
        print("\nThe Convolutional autoencoder produces more discriminable embeddings.")
    
    # Save analysis to file
    with open(f'{output_dir}/discriminability_analysis.txt', 'w') as f:
        f.write("GRU Autoencoder Analysis:\n")
        f.write(f"Average intra-class distance: {gru_analysis['avg_intra_class_distance']:.4f}\n")
        f.write(f"Average inter-class distance: {gru_analysis['avg_inter_class_distance']:.4f}\n")
        f.write(f"Discriminability ratio: {gru_analysis['discriminability_ratio']:.4f}\n\n")
        
        f.write("Convolutional Autoencoder Analysis:\n")
        f.write(f"Average intra-class distance: {conv_analysis['avg_intra_class_distance']:.4f}\n")
        f.write(f"Average inter-class distance: {conv_analysis['avg_inter_class_distance']:.4f}\n")
        f.write(f"Discriminability ratio: {conv_analysis['discriminability_ratio']:.4f}\n\n")
        
        if gru_analysis['discriminability_ratio'] > conv_analysis['discriminability_ratio']:
            f.write("The GRU autoencoder produces more discriminable embeddings.\n")
        else:
            f.write("The Convolutional autoencoder produces more discriminable embeddings.\n")
    
    print(f"\nResults saved to {output_dir}")
    return gru_plot_path, conv_plot_path, f'{output_dir}/discriminability_analysis.txt'

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="t-SNE visualization of autoencoder embeddings.")
    parser.add_argument("--assignment_dir", type=str, default=".", help="Path to the assignment directory.", required=True)

    # Add the assignment directory to the path
    args = parser.parse_args()
    sys.path.append(args.assignment_dir)

    main()
