import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def get_celeba_loader(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 2
) -> DataLoader:
    """
        CelebA Loader
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=1), # VanillaGAN we will use grayscale only
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # scale to [-1, 1] to match tanh output
    ])

    dataset = datasets.ImageFolder(root=root, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True, # faster CPU to GPU transfer
        drop_last=True, # avoid batch-size of 1 edge case in BatchNorm
    )

class FakeDataset(Dataset):
    """
        Local mock dataset to avoid downloading full data
        Produces random tensors shaped exactly like real CelebA 32x32 grayscale
    """
    def __init__(
        self, 
        num_samples = 1000,
        image_size = 32,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Shape: (1, H, W), range [-1, 1]
        return torch.randn(1, self.image_size, self.image_size), 0 # 0 = dummy label

def get_fake_loader(batch_size, image_size=32):
    dataset = FakeDataset(num_samples=1000, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
