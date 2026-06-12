import torch
import torch.nn as nn

NUM_CLASSES = 5

def weights_init(m):
    classname = m.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim=100, features_g=64, num_classes=NUM_CLASSES):
        super().__init__()

        # projects label vector → (B, num_classes, 1, 1)
        # so it can be concatenated with z on channel dim
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            self._block(z_dim + num_classes, features_g * 8, kernel=4, stride=1, padding=0),
            self._block(features_g * 8, features_g * 4, kernel=4, stride=2, padding=1),
            self._block(features_g * 4, features_g * 2, kernel=4, stride=2, padding=1),
            self._block(features_g * 2, features_g,     kernel=4, stride=2, padding=1),
            nn.ConvTranspose2d(features_g, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, labels):
        emb = self.label_embedding(labels) # (B, num_classes)
        emb = emb.unsqueeze(-1).unsqueeze(-1) # (B, num_classes, 1, 1)
        x = torch.cat([x, emb], dim=1) # (B, z_dim + num_classes, 1, 1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, features_d=64, num_classes=NUM_CLASSES):
        super().__init__()

        # projects label vector → (B, num_classes, 64, 64)
        # tiled spatially to match image dims for concat on channel dim
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, 64 * 64),
            nn.ReLU()
        )

        # first conv takes 3 + num_classes input channels
        self.net = nn.Sequential(
            # no BatchNorm on input
            nn.Conv2d(3 + num_classes, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(features_d,     features_d * 2, kernel=4, stride=2, padding=1),
            self._block(features_d * 2, features_d * 4, kernel=4, stride=2, padding=1),
            self._block(features_d * 4, features_d * 8, kernel=4, stride=2, padding=1),

            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, labels):
        emb = self.label_embedding(labels) # (B, 64 * 64)
        emb = emb.view(x.size(0), 1, 64, 64) # (B, 1, 64, 64)
        emb = emb.expand(-1, NUM_CLASSES, -1, -1) # (B, num_classes, 64, 64)
        x = torch.cat([x, emb], dim=1) # (B, 3 + num_classes, 64, 64)
        return self.net(x).view(x.size(0), -1) # (B, 1)

# sanity check
if __name__ == "__main__":
    G = Generator(z_dim=100, features_g=64)
    D = Discriminator(features_d=64)

    z      = torch.randn(4, 100, 1, 1)
    labels = torch.randint(0, 2, (4, 5)).float()

    fake = G(z, labels)
    out  = D(fake, labels)

    print("G output:", fake.shape)    # (4, 3, 64, 64)
    print("D output:", out.shape)     # (4, 1)
