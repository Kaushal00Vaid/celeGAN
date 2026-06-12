from dataset import get_fake_loader
from model import Generator, Discriminator, weight_init
import torch

# fake loader with RGB 64x64
loader = get_fake_loader(batch_size=4, image_size=64)

# update FakeDataset to output 3 channels first:
# return torch.randn(3, self.image_size, self.image_size), 0

G = Generator(z_dim=100, features_g=64)
D = Discriminator(features_d=64)
G.apply(weight_init)
D.apply(weight_init)

real, _ = next(iter(loader))
real = real          # (4, 3, 64, 64) — no flatten

noise = torch.randn(4, 100, 1, 1)
fake = G(noise)
out = D(fake)

print("real shape:", real.shape)    # (4, 3, 64, 64)
print("fake shape:", fake.shape)    # (4, 3, 64, 64)
print("D output:", out.shape)       # (4, 1)