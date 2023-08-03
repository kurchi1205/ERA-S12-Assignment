from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch

class CIFAR10WithAlbumentations(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label
    
class CIFAR10LitModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2470, 0.2435, 0.2616]

        self.train_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.PadIfNeeded(min_height=4, min_width=4, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                ToTensorV2(),
            ]
        )

        self.test_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )

    def setup(self, stage: str = "train"):
        self.train_ds = CIFAR10WithAlbumentations('./data', train=True, download=True, transform=self.train_transforms)
        self.test_ds = CIFAR10WithAlbumentations('./data', train=False, download=True, transform=self.test_transforms)
        self.val_ds = CIFAR10WithAlbumentations('./data', train=False, download=True, transform=self.test_transforms)


    def train_dataloader(self):
        cuda = torch.cuda.is_available()
        dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
        return DataLoader(self.train_ds, **dataloader_args)

    def val_dataloader(self):
        cuda = torch.cuda.is_available()
        dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
        return DataLoader(self.val_ds, **dataloader_args)

    def test_dataloader(self):
        cuda = torch.cuda.is_available()
        dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
        return DataLoader(self.test_ds, **dataloader_args)
