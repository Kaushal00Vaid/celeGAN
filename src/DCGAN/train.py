import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Discriminator, Generator, weight_init
from dataset import get_fake_loader, get_tensor_loader
from utils import save_image_grid, plot_loss_curve

# hyperparameters
IMAGE_SIZE = 64
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 2e-4
Z_DIM = 100
FEATURES_G = 64
FEATURES_D = 64
IMAGE_DIM = 64 * 64 * 3 # 12288
NUM_EPOCHS = 50
SAVE_EPOCHS = [1, 5, 10, 25, 50]
IS_KAGGLE = torch.cuda.is_available()
PT_PATH = "/kaggle/input/celeba-all-phases/celeba_64x64_hr.pt"

def train():
    # data
    if IS_KAGGLE and PT_PATH:
        loader = get_tensor_loader(pt_path=PT_PATH, batch_size=BATCH_SIZE)
    else:
        loader = get_fake_loader(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    # model
    G = Generator(z_dim=Z_DIM, features_g=FEATURES_G).to(DEVICE)
    D = Discriminator(features_d=FEATURES_D).to(DEVICE)
    G.apply(weight_init)
    D.apply(weight_init)

    # optims
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # loss
    criterion = nn.BCELoss()

    # fixed noise
    fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(DEVICE)

    loss_G_history = []
    loss_D_history = []

    # training
    print("--------------- Starting training ---------------")

    for epoch in range(1, NUM_EPOCHS+1):
        loss_G_epoch = 0.0
        loss_D_epoch = 0.0

        for batch_idx, (real,) in enumerate(tqdm(loader, desc=f"Epoch [{epoch}/{NUM_EPOCHS}]", leave=True)):
            
            real = real.to(DEVICE) # shape stays (B, 3, 64, 64)
            B = real.size(0)

            # train D
            noise = torch.randn(B, Z_DIM, 1, 1).to(DEVICE)
            fake = G(noise).detach()

            real_labels = torch.ones(B, 1).to(DEVICE)
            fake_labels = torch.zeros(B, 1).to(DEVICE)

            loss_D_real = criterion(D(real), real_labels)
            loss_D_fake = criterion(D(fake), fake_labels)

            loss_D = loss_D_fake + loss_D_real

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # train G
            noise = torch.randn(B, Z_DIM, 1, 1).to(DEVICE)
            fake = G(noise)

            loss_G = criterion(D(fake), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_D_epoch += loss_D.item()
            loss_G_epoch += loss_G.item()

        # end of epoch
        avg_G = loss_G_epoch / len(loader)
        avg_D = loss_D_epoch / len(loader)
        loss_G_history.append(avg_G)
        loss_D_history.append(avg_D)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | loss_D: {avg_D:.4f} | loss_G: {avg_G:.4f}")

        if epoch in SAVE_EPOCHS:
            with torch.no_grad():
                sample = G(fixed_noise)
                save_image_grid(sample, epoch, path="outputs/")

    plot_loss_curve(loss_G_history, loss_D_history, path="outputs/")


if __name__ == "__main__":
    train()
