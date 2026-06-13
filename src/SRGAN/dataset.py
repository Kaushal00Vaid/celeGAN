import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import random

def get_srgan_tensor_loader(
    lr_pt_path,
    hr_pt_path,
    batch_size,
):
    lr_tensor = torch.load(lr_pt_path, weights_only=True)
    hr_tensor = torch.load(hr_pt_path, weights_only=True)

    assert lr_tensor.shape[0] == hr_tensor.shape[0], \
        f"LR/HR sample count mismatch: {lr_tensor.shape[0]} vs {hr_tensor.shape[0]}"

    dataset = TensorDataset(lr_tensor, hr_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

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
    tensor = torch.load(pt_path, weights_only=True)
    dataset = TensorDataset(tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

class FakeDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=64, num_classes=5):
        super().__init__()

        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        labels = torch.randint(0, 2, (self.num_classes,)).float()

        return image, labels 


def get_fake_loader(batch_size, image_size=64, num_classes=5):
    dataset = FakeDataset(num_samples=1000, image_size=image_size, num_classes=num_classes)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )


# sanity check
loader = get_srgan_tensor_loader(
    lr_pt_path="dataset\celeba_16x16_lr.pt",
    hr_pt_path="dataset\celeba_64x64_hr.pt",
    batch_size=4
)

lr, hr = next(iter(loader))
print("LR:", lr.shape)    # (4, 3, 16, 16)
print("HR:", hr.shape)    # (4, 3, 64, 64)
print("LR range:", lr.min().item(), lr.max().item())   # ~(-2, 2) normalized
print("HR range:", hr.min().item(), hr.max().item())   # ~(-2, 2) normalized