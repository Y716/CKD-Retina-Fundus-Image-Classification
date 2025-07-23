from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import lightning.pytorch as pl

class FundusDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=32, img_size=224, num_workers=4):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers )
