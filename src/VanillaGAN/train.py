import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Discriminator, Generator
from dataset import get_celeba_loader, get_fake_loader, get_tensor_loader
from utils import save_image_grid, plot_loss_curve

# hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 2e-4
BATCH_SIZE = 128
Z_DIM = 100
IMAGE_DIM = 32 * 32 * 1 # 1024 - 32x32 grayscale
NUM_EPOCHS = 50
SAVE_EPOCHS = [1, 5, 10, 25, 50] # saving image grids to view the learning of G
IS_KAGGLE = torch.cuda.is_available()
PT_PATH=""

def train():
    # data
    if IS_KAGGLE and PT_PATH:
        loader = get_tensor_loader(pt_path=PT_PATH, batch_size=BATCH_SIZE)
    elif IS_KAGGLE:
        loader = get_celeba_loader(root=ROOT, image_size=32, batch_size=BATCH_SIZE)
    else:
        loader = get_fake_loader(batch_size=BATCH_SIZE, image_size=32)

    # models
    D = Discriminator(img_dim=IMAGE_DIM).to(DEVICE)
    G = Generator(z_dim=Z_DIM, img_dim=IMAGE_DIM).to(DEVICE)

    # loss + optims
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    # betas=(0.5, 0.999) is GAN standard (experimentally proven) - 0.9 causes oscillation

    # fixed noise: same vector every save -> comparable grids across epochs
    fixed_noise = torch.randn(64, Z_DIM).to(DEVICE)

    loss_G_history = []
    loss_D_history = []

    # training
    print("--------------- Starting training ---------------")

    for epoch in range(1, NUM_EPOCHS + 1):
        loss_G_epoch = 0.0
        loss_D_epoch = 0.0

        for batch_idx, (real, _) in enumerate(tqdm(loader, desc=f"Epoch [{epoch}/{NUM_EPOCHS}]", leave=True)):

            real = real.view(real.size(0), -1).to(DEVICE) # flatten: (B, 1, 32, 32) --> (B, 1024)
            B = real.size(0)

            # train D
            """
                1. Get real img from datalaoder
                2. Generate fake images (G, no grad)
                3. D forward pass on real --> loss_real
                4. D forward pass on fake --> loss_fake
                5. loss_D = loss_real + loss_fake
                6. Backprop + wt update
            """
            noise = torch.randn(B, Z_DIM).to(DEVICE)
            fake = G(noise).detach() # no G gradients here

            real_labels = torch.ones(B, 1).to(DEVICE)
            fake_labels = torch.zeros(B, 1).to(DEVICE)

            loss_D_real = criterion(D(real), real_labels)
            loss_D_fake = criterion(D(fake), fake_labels)
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # train G
            """
                1. Generate NEW fake images (fresh noise)
                2. D forward pass on fake (no D grad update)
                3. loss_G = how badly D was fooled
                4. backprop and wtt update
            """
            noise = torch.randn(B, Z_DIM).to(DEVICE)
            fake = G(noise)

            # G wants D to output 1 (real) for its fakes
            loss_G = criterion(D(fake), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_G_epoch += loss_G.item()
            loss_D_epoch += loss_D.item()

        # end of epoch
        avg_G = loss_G_epoch / len(loader)
        avg_D = loss_D_epoch / len(loader)
        loss_G_history.append(avg_G)
        loss_D_history.append(avg_D)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | loss_D: {avg_D:.4f} | loss_G: {avg_G:.4f}")

        if epoch in SAVE_EPOCHS:
            with torch.no_grad():
                sample = G(fixed_noise) # (64, 1024)
                sample = sample.view(-1, 1, 32, 32) # reshape for grid
                save_image_grid(sample, epoch, path="outputs/")

    plot_loss_curve(loss_G_history, loss_D_history, path="outputs/")


if __name__ == "__main__":
    train()
