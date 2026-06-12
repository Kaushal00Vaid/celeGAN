import torch
import torch.nn as nn

def weight_init(m):
    """
        DCGAN paper: init Conv and BatchNorm weights from Normal(0, 0.02) and Normal (1.0, 0.02) respectively.
        Called once on G and D after instantiation.
    """
    classname = m.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
        Input: (B, z_dim, 1, 1)
        Output: (B, 3, 64, 64)
    """

    def __init__(self, z_dim=100, features_g=64):
        super().__init__()

        # features_g is the base channel multiplier 
        self.net = nn.Sequential(
            # (B, z_dim, 1, 1) --> (B, features_g*8, 4, 4)
            self._block(z_dim, features_g * 8, kernel=4, stride=1, padding=0),

            # (B, fg*8, 4, 4) --> (B, fg*4, 8, 8)
            self._block(features_g * 8, features_g * 4, kernel=4, stride=2, padding=1),

            # (B, fg*4, 8, 8) --> (B, fg*2, 16, 16)
            self._block(features_g * 4, features_g * 2, kernel=4, stride=2, padding=1),

            # (B, fg*2, 16, 16) --> (B, fg, 32, 32)
            self._block(features_g * 2, features_g, kernel=4, stride=2, padding=1),

            # output layer -- no batchNorm
            # (B, fg, 32, 32) --> (B, 3, 64, 64)
            nn.ConvTranspose2d(features_g, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # G uses ReLu and not LeakyReLU
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
        Input: (B, 3, 64, 64)
        Output: (B, 1, 1, 1) --> squeezed to (B, 1)
    """

    def __init__(self, features_d=64):
        super().__init__()

        self.net = nn.Sequential(
            # (B, 3, 64, 64) --> (B, fd, 32, 32)
            nn.Conv2d(3, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, fd, 32, 32) --> (B, fd*2, 16, 16)
            self._block(features_d, features_d * 2, kernel=4, stride=2, padding=1),

            # (B, fd*2, 16, 16) --> (B, fd*4, 8, 8)
            self._block(features_d * 2, features_d * 4, kernel=4, stride=2, padding=1),

            # (B, fd*4, 8, 8) --> (B, fd*8, 4, 4)
            self._block(features_d * 4, features_d * 8, kernel=4, stride=2, padding=1),

            # output layer
            # (B, fd*8, 4, 4) --> (B, 1, 1, 1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True), # D uses LeakyReLU
        )
    
    def forward(self, x):
        return self.net(x).view(x.size(0), -1) # (B, 1, 1, 1) --> (B, 1)

# sanity check
if __name__ == "__main__":
    G = Generator(z_dim=100, features_g=64)
    D = Discriminator(features_d=64)

    z = torch.randn(4, 100, 1, 1)
    fake = G(z)
    out = D(fake)

    print("G output:", fake.shape)   # expect (4, 3, 64, 64)
    print("D output:", out.shape)    # expect (4, 1)
    print("D range:", out.min().item(), out.max().item())  # expect ~(0, 1)
