import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

def save_image_grid(fake_samples, epoch, path="outputs/"):
    """
        Saves a grid of generated images to disk.
        fake_samples: (B, 1, H, W) tensor in [-1, 1]
    """

    os.makedirs(path, exist_ok=True)

    # denormalise for display
    fake_samples = (fake_samples + 1) / 2

    grid = vutils.make_grid(fake_samples, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"epoch_{epoch:03d}.png"), dpi=100)
    plt.close()

def plot_loss_curve(loss_G, loss_D, path="outputs/"):
    """
        Saves G and D loss curves across epochs.
    """
    os.makedirs(path, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(loss_G, label="Generator Loss", color="steelblue")
    plt.plot(loss_D, label="Discriminator Loss", color="tomato")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, "loss_curve.png"), dpi=100)
    plt.close()

def load_and_display_progression(output_dir, epochs):
    """
        Notebook-only function.
        Loads saved grids from disk and plots them side by side for visual progression.
        epochs: e.g. [1, 5, 10, 25, 50]
    """
    fig, axes = plt.subplots(1, len(epochs), figsize=(4 * len(epochs), 4))

    for ax, epoch in zip(axes, epochs):
        img_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
        img = plt.imread(img_path)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Epoch {epoch}")
        ax.axis("off")

    plt.suptitle("GAN Progression", fontsize=14)
    plt.tight_layout()
    plt.show()