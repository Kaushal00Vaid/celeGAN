from model import Generator, Discriminator, VGGFeatureExtractor
import torch

DEVICE = torch.device("cpu")
B = 2

G   = Generator(num_res_blocks=4).to(DEVICE)   # 4 not 16 for speed
D   = Discriminator().to(DEVICE)
vgg = VGGFeatureExtractor().to(DEVICE)
vgg.eval()

lr = torch.randn(B, 3, 16, 16)
hr = torch.randn(B, 3, 64, 64)

fake_hr       = G(lr)
d_out         = D(fake_hr)
feats_real    = vgg(hr)
feats_fake    = vgg(fake_hr)

loss_adv = torch.nn.BCELoss()(d_out, torch.ones(B, 1))
loss_per = torch.nn.MSELoss()(feats_fake, feats_real)
loss_mse = torch.nn.MSELoss()(fake_hr, hr)
loss_G   = loss_per + 1e-3 * loss_adv + 1e-2 * loss_mse

print("fake_hr:", fake_hr.shape)      # (2, 3, 64, 64)
print("d_out:", d_out.shape)          # (2, 1)
print("feats:", feats_fake.shape)     # (2, 256, 16, 16)
print("loss_G:", loss_G.item())       # any finite number
print("loss_adv:", loss_adv.item())
print("loss_per:", loss_per.item())
print("loss_mse:", loss_mse.item())