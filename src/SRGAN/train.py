import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import Discriminator, Generator, VGGFeatureExtractor
from dataset import get_srgan_tensor_loader
from utils import save_image_grid, plot_loss_curve

# hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4 # lower thaan other -- ResNet is sensitive
BATCH_SIZE = 64 # memory heavy
NUM_EPOCHS = 50
SAVE_EPOCHS = [1, 5, 10, 25, 50]

LR_PT_PATH = ""
HR_PT_PATH = ""

# loss weights
LAMBDA_PERCEPTUAL = 1.0
LAMBDA_ADV = 1e-3
LAMBDA_MSE = 1e-2

def train():
    # data
    loader = get_srgan_tensor_loader(
        lr_pt_path=LR_PT_PATH,
        hr_pt_path=HR_PT_PATH,
        batch_size=BATCH_SIZE
    )

    # model
    G = Generator(num_res_blocks=16).to(DEVICE)
    D = Discriminator().to(DEVICE)
    vgg = VGGFeatureExtractor().to(DEVICE)
    vgg.eval()

    # loss functions
    criterion_adv = nn.BCELoss()
    criterion_mse = nn.MSELoss() # pixel loss
    criterion_per = nn.MSELoss() # perceptual loss - MSE on VGG feats

    # optims
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # fixed samples for visualization
    fixed_lr, fixed_hr = next(iter(loader))
    fixed_lr = fixed_lr[:16].to(DEVICE) # show 16 samples per grid
    fixed_hr = fixed_hr[:16]

    loss_G_history = []
    loss_D_history = []

    print("--------------- Starting training ---------------")

    for epoch in range(1, NUM_EPOCHS + 1):
        loss_G_epoch = 0.0
        loss_D_epoch = 0.0

        for batch_idx, (lr, hr) in enumerate(tqdm(loader, desc=f"Epoch [{epoch}/{NUM_EPOCHS}]", leave=True)):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            B = lr.size(0)

            real_labels = torch.ones(B, 1).to(DEVICE)
            fake_labels = torch.zeros(B, 1).to(DEVICE)

            # train D
            fake_hr = G(lr).detach()

            loss_D_real = criterion_adv(D(hr), real_labels)
            loss_D_fake = criterion_adv(D(fake_hr), fake_labels)
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # train G
            fake_hr = G(lr)

            # adversarial loss: food D
            loss_adv = criterion_adv(D(fake_hr), real_labels)
            
            # perceptual: VGG feat similarity
            with torch.no_grad():
                features_real = vgg(hr)
            features_fake = vgg(fake_hr)
            loss_perceptual = criterion_per(features_fake, features_real)

            # pixel: coarse structure
            loss_mse = criterion_mse(fake_hr, hr)

            # combined: G loss
            loss_G = (LAMBDA_PERCEPTUAL * loss_perceptual
                    + LAMBDA_ADV * loss_adv
                    + LAMBDA_MSE * loss_mse
            )

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
                fake_hr = G(fixed_lr)
                # save LR, fake HR, real HR side by side for comparison
                save_image_grid(fixed_lr,  epoch, path="outputs/lr/")
                save_image_grid(fake_hr,   epoch, path="outputs/fake_hr/")
                save_image_grid(fixed_hr[:16].to(DEVICE), epoch, path="outputs/real_hr/")

    plot_loss_curve(loss_G_history, loss_D_history, path="outputs/")


if __name__ == "__main__":
    train()
