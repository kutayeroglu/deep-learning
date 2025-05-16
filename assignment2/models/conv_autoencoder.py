import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 28, 28)
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=0),  # 3x3 (latent space)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),  # 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, 3, stride=2, padding=1, output_padding=1
            ),  # 28x28
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Reshape input if needed (batch, 1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)  # Remove channel dim to match original input shape
