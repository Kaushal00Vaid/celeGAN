import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import transforms, datasets
import random

def get_celeba_loader(
    root,
    image_size,
    batch_size,
    num_workers=0,
    max_samples=20000
):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    full_dataset = datasets.ImageFolder(root=root, transform=transform)

    indices = random.sample(range(len(full_dataset)), min(max_samples, len(full_dataset)))
    dataset = Subset(full_dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

def get_tensor_loader(pt_path, batch_size):
    tensor = torch.load(pt_path, weights_only=True) # shape: (N, C, H, W)
    dataset = TensorDataset(tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

class FakeDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=64):
        super().__init__()

        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(3, self.image_size, self.image_size), 0


def get_fake_loader(batch_size, image_size=64):
    dataset = FakeDataset(num_samples=1000, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)