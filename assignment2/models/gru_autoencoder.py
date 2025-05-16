import torch.nn as nn


# TODO: BiGRUAutoencoder
class GRUAutoencoder(nn.Module):
    def __init__(self, input_size=28, hidden_dim=128, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Encode: Process each row (time step)
        _, hidden = self.encoder(x)  # hidden: (num_layers, batch, hidden_size)

        # Prepare decoder input (learned from hidden state)
        latent = hidden[-1].unsqueeze(0)  # Use last layer's hidden state
        decoder_input = latent.repeat(seq_len, 1, 1).permute(
            1, 0, 2
        )  # "dummy" seq: each seq is same latent

        # Decode: Reconstruct sequence
        decoder_output, _ = self.decoder(decoder_input, hidden)
        reconstructed = self.fc(decoder_output)

        return reconstructed

    def encode(self, x):
        # Extract the encoder's final hidden state
        _, hidden = self.encoder(x)  # hidden: (num_layers, batch_size, hidden_dim)
        return hidden[-1]  # Return last layer's hidden state (batch_size, hidden_dim)
