from dataset import get_fake_loader
from model import Generator, Discriminator, weights_init
import torch


B = 4
NUM_CLASSES = 5
Z_DIM = 100

# loader = get_fake_loader(batch_size=B, num_classes=NUM_CLASSES)

G = Generator(z_dim=Z_DIM, features_g=64, num_classes=NUM_CLASSES)
D = Discriminator(features_d=64, num_classes=NUM_CLASSES)
G.apply(weights_init)
D.apply(weights_init)

real   = torch.randn(B, 3, 64, 64)
labels = torch.randint(0, 2, (B, NUM_CLASSES)).float()
noise  = torch.randn(B, Z_DIM, 1, 1)

fake        = G(noise, labels).detach()
loss_D_real = torch.nn.BCELoss()(D(real, labels),  torch.ones(B, 1))
loss_D_fake = torch.nn.BCELoss()(D(fake, labels), torch.zeros(B, 1))
loss_D      = loss_D_real + loss_D_fake

fake   = G(noise, labels)
loss_G = torch.nn.BCELoss()(D(fake, labels), torch.ones(B, 1))

print("loss_D:", loss_D.item())    # expect ~1.0–2.0
print("loss_G:", loss_G.item())    # expect ~0.5–2.0
print("fake shape:", fake.shape)   # (4, 3, 64, 64)