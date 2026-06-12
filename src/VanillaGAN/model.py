import torch
import torch.nn as nn

class Generator(nn.Module):
    """
        Maps a random noise vector (latent space) -> flattened image.
        z_dim : size of the input noise vector (typically 100)
        img_dim : H * W * C = 32 * 32 * 1 = 1024 for grayscale 32x32
    """

    def __init__(self, z_dim=100, img_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            # block 1: z_dim --> 256
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),

            # block 2: 256 --> 512
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            # block 3: 512 --> 1024
            nn.Linear(512, 1024),
            nn.LeakyReLU(),

            # output: 1024 --> img_dim, squash to [-1, 1]
            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    """
        Maps a flattened image --> scalar probability of being real.
        img_dim : must match Generator's img_dim
    """

    def __init__(self, img_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            # block 1 --> img_dim --> 512
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),

            # block 2: 512 --> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            # output: 256 --> single logit -- real or fake
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)
