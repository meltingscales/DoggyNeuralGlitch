import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2),  # outputs mean and logvar concatenated
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 64, channels: list[int] = None):
        super().__init__()
        if channels is None:
            channels = [64, 32, 16, 8, 3]

        # Project latent vector to 16-channel 16x16 spatial base
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.ReLU(),
        )

        # Build convolutional upsampling layers
        # Each ConvTranspose2d(k, stride=2, padding=1) doubles spatial dims
        layers = []
        in_ch = 16
        for i, out_ch in enumerate(channels):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if i < len(channels) - 1:
                layers.append(nn.ReLU())
            in_ch = out_ch
        layers.append(nn.Tanh())
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 16, 16, 16)  # (batch, 16 channels, 16x16)
        return self.conv(h)          # -> (batch, 3, 512, 512)


class TinyVAE(nn.Module):
    def __init__(self, input_dim: int = 128, latent_dim: int = 64,
                 channels: list[int] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight, gain=2.0)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=0.1)

    def reparameterize(self, mean: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar
