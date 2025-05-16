import numpy as np
import torch


def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, lbls = batch
            inputs = inputs.to(device)

            # Get embeddings (batch_size, hidden_dim)
            latent = model.encode(inputs)

            embeddings.append(latent.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    return np.concatenate(embeddings), np.concatenate(labels)
