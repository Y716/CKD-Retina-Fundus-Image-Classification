import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, image_path, mask_path, augment=False):
        self.images_path = sorted(image_path)
        self.masks_path = sorted(mask_path)
        self.augment = augment

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=(-45, 45)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
        self.normalize = transforms.Normalize([0.517840866388059, 0.26527648551791333, 0.1602001751170439], [0.3529104350988155, 0.1800796823570141, 0.1031302883915354])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        MAX_INT_32 = 2**3
        seed = random.randint(-MAX_INT_32, MAX_INT_32)
        
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        normal_image = self.normalize(image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.transform(mask)
        mask = torch.where(mask >= .5, 1, 0)

        return image, normal_image, mask
