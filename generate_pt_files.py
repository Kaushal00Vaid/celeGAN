"""
    Instructions to run
    Add public kaggle dataset to your kaggle notebook

    Run this cell. It will create 4 pt files.
    1. celeba_32x32_gray.pt (32x32 grayscale)
    2. celeba_64x64_rgb.pt (64x64 rgb)
    3. celeba_64x64_hr.pt (64x64 rgb)
    4. celeba_16x16_lr.pt (16x16 rgb)

    Download the 1st, 3rd and 4th
    (2nd is not needed)

    And then add it to dataset/ locally or upload to kaggle datasets
"""

# grayscale 32x32 → Phase 1 (Vanilla GAN)
transform_gray = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# RGB 64x64 → Phase 2 (DCGAN), Phase 3 (cGAN)
transform_rgb_64 = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# RGB 64x64 → Phase 4 (SRGAN) high-res target
# RGB 16x16 → Phase 4 (SRGAN) low-res input
transform_rgb_16 = transforms.Compose([
    transforms.Resize(16),
    transforms.CenterCrop(16),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

import torch
from torchvision import datasets
from tqdm import tqdm

NUM_SAMPLES = 50000
root = "/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba"

for name, transform in [
    ("celeba_32x32_gray",  transform_gray),
    ("celeba_64x64_rgb",   transform_rgb_64),
    ("celeba_64x64_hr",    transform_rgb_64),   # SRGAN high-res target
    ("celeba_16x16_lr",    transform_rgb_16),   # SRGAN low-res input
]:
    dataset = datasets.ImageFolder(root=root, transform=transform)
    images = []
    for i in tqdm(range(NUM_SAMPLES), desc=name):
        img, _ = dataset[i]
        images.append(img)
    tensor = torch.stack(images)
    torch.save(tensor, f"{name}.pt")
    print(f"{name}: {tensor.shape}")
