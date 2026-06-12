import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import random
import pandas as pd

SELECTED_ATTRS = [
    "Smiling", 
    "Male", 
    "Young", 
    "Eyeglasses", 
    "Black_Hair",
]

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

def get_cgan_tensor_loader(pt_path, attr_csv, batch_size):
    images = torch.load(pt_path, weights_only=True) # (50000, 3, 64, 64)
    N = images.shape[0]

    df = pd.read_csv(attr_csv)

    df = df[SELECTED_ATTRS].iloc[:N].replace(-1, 0)
    labels = torch.tensor(df.values, dtype=torch.float32) # (50000, 5)

    print(f"Images: {images.shape}, Labels: {labels.shape}")

    dataset = TensorDataset(images, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

# sanity check
loader = get_cgan_tensor_loader(
    pt_path="dataset\celeba_64x64_hr.pt",
    attr_csv="dataset\list_attr_celeba.csv",
    batch_size=4
)

images, labels = next(iter(loader))
print("images:", images.shape)   # (4, 3, 64, 64)
print("labels:", labels.shape)   # (4, 5)
print("label sample:", labels[0])  # e.g. tensor([1., 0., 1., 0., 1.])
print("label range:", labels.min().item(), labels.max().item())  # 0.0, 1.0

