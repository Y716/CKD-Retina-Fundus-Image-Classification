import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import lightning.pytorch as pl
import torch
from PIL import Image
import numpy as np
from rich.console import Console

console = Console()

class FundusDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=32, img_size=224, num_workers=4, 
                 use_augmentation=True):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        self.class_names = None
        self.class_counts = None

    def setup(self, stage=None):
        # Define training transforms with augmentation
        if self.use_augmentation:
            self.train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Define validation transforms (no augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transforms)
        
        # Store class names and counts
        self.class_names = self.train_dataset.classes
        self.class_counts = self._get_class_counts()
        
        # Print dataset information
        self._print_dataset_info()

    def _get_class_counts(self):
        """Count samples per class in the training and validation sets"""
        train_counts = {cls: 0 for cls in self.class_names}
        val_counts = {cls: 0 for cls in self.class_names}
        
        # Count training samples
        for _, label in self.train_dataset.samples:
            class_name = self.class_names[label]
            train_counts[class_name] += 1
            
        # Count validation samples
        for _, label in self.val_dataset.samples:
            class_name = self.class_names[label]
            val_counts[class_name] += 1
            
        return {"train": train_counts, "val": val_counts}

    def _print_dataset_info(self):
        """Print information about the dataset"""
        console.print("[bold cyan]Dataset Information:[/bold cyan]")
        console.print(f"Image size: {self.img_size}x{self.img_size}")
        console.print(f"Number of classes: {len(self.class_names)}")
        console.print(f"Class names: {', '.join(self.class_names)}")
        console.print(f"Total training samples: {len(self.train_dataset)}")
        console.print(f"Total validation samples: {len(self.val_dataset)}")
        
        console.print("[bold cyan]Class distribution:[/bold cyan]")
        for cls in self.class_names:
            train_count = self.class_counts["train"][cls]
            val_count = self.class_counts["val"][cls]
            train_pct = 100 * train_count / len(self.train_dataset)
            val_pct = 100 * val_count / len(self.val_dataset)
            
            console.print(f"  {cls}: {train_count} train ({train_pct:.1f}%), {val_count} val ({val_pct:.1f}%)")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True)
            
    def get_class_weights(self):
        """Calculate class weights inversely proportional to class frequencies"""
        if self.class_counts is None:
            return None
            
        # Get counts from training set
        counts = np.array([self.class_counts["train"][cls] for cls in self.class_names])
        weights = 1.0 / counts
        weights = weights / np.sum(weights) * len(counts)  # normalize
        
        return torch.tensor(weights, dtype=torch.float32)