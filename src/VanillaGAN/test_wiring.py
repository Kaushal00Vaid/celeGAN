# confirms your wiring is correct before touching Kaggle

from dataset import get_fake_loader
from model import Generator, Discriminator
import torch

loader = get_fake_loader(batch_size=4, image_size=32)
real, _ = next(iter(loader))
print("real shape:", real.shape)          # expect (4, 1, 32, 32)

G = Generator(z_dim=100, img_dim=1024)
D = Discriminator(img_dim=1024)

noise = torch.randn(4, 100)
fake = G(noise)
print("fake shape:", fake.shape)          # expect (4, 1024)

out = D(fake)
print("D output shape:", out.shape)       # expect (4, 1)
print("D output range:", out.min().item(), out.max().item())  # expect ~(0,1)