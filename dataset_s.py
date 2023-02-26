import os
from pathlib import Path
import cv2
import hydra
import pandas as pd
import torch
import torchvision.transforms as tvf
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from kornia.morphology import opening, closing

class CustomDataset(Dataset):
    def __init__(self, data_dir, augment=True, augment_config=False, crop_size=256):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.input_size = crop_size
        self.transform = get_image_transformer(self.input_size, augment, augment_config)
        self.dataset = ImageFolder(root=self.data_dir)

    def __getitem__(self,index):
        img_path = self.dataset.samples[index][0]
        label = self.dataset.samples[index][1]
        img = Image.open(img_path).convert('RGB')
        
        img_tensor = self.transform(img)
        return img_tensor, label, img_path

    def __len__(self):
        return len(self.dataset)


class CustomDataModule(LightningDataModule):
    def __init__(self,config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            self.config.train_data_dir,
            augment=self.config.train_augment,
            crop_size=self.config.crop_size,
            augment_config=self.config.augmentation,
        )

        self.val_dataset = CustomDataset(
            self.config.val_data_dir,
            augment=self.config.val_augment,
            crop_size=self.config.crop_size,
            augment_config=self.config.augmentation,

        )
        self.test_dataset = CustomDataset(
            self.config.test_data_dir,
            augment=self.config.test_augment,
            crop_size=self.config.crop_size,
            augment_config=self.config.augmentation,

        )
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,)
class OpeningAndClosing(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, tensor):
        kernel = torch.ones(3, 3)
        gray = tvf.functional.rbg_to_grayscale()
        tensor = tensor.unsqueeze(0)
        opened_img = opening(tensor, kernel)
        closed_img = closing(opened_img, kernel)
        closed_img = closed_img.squeeze()
        return closed_img


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
    
def get_image_transformer(input_size,augment, augment_config=None):
    transforms = []
    if augment:
        transforms += [tvf.Resize([input_size, input_size])]
        for aug in augment_config:
            transforms += [hydra.utils.instantiate(augment_config[aug])]
        transforms += [
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             OpeningAndClosing(),
        ]
    else:
        transforms += [
            tvf.Resize([input_size, input_size]),
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             OpeningAndClosing(),
        ]
    transforms = tvf.Compose(transforms)
    return transforms