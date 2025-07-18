import albumentations as A
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import glob
from sklearn.preprocessing import LabelEncoder
import os

medical_transforms = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.SquareSymmetry(p=0.5),  # All 8 rotations/flips - proper for medical data
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # Tissue-like distortion
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Center around 0
    A.ToTensorV2(),
])

class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Gagal membaca gambar: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(self.labels[idx], dtype=torch.long)

def load_image_paths_and_labels(folder):
    class_dirs = sorted(os.listdir(folder))  # urutkan supaya label konsisten
    image_paths = []
    labels = []

    for label_name in class_dirs:
        class_path = os.path.join(folder, label_name)
        if not os.path.isdir(class_path):
            continue
        images = glob.glob(os.path.join(class_path, '*'))
        image_paths.extend(images)
        labels.extend([label_name] * len(images))

    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    return image_paths, label_ids, le.classes_  # image paths, integer labels, class names

train_paths, train_labels, class_names = load_image_paths_and_labels('split_dataset/train')
val_paths, val_labels, _ = load_image_paths_and_labels('split_dataset/test')
num_classes = len(class_names)

print(f"Detected classes: {class_names}")
print(f"Number of train samples: {len(train_paths)}")
print(f"Number of val samples: {len(val_paths)}")


# Create datasets with ImageNet-style transforms
train_dataset = ImageClassificationDataset(train_paths, train_labels, medical_transforms)
val_dataset = ImageClassificationDataset(val_paths, val_labels, medical_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Quick check - load one image and see the output
sample_image = cv2.imread('split_dataset/train/normal/NL_001.png')
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

augmented = medical_transforms(image=sample_image)
print(f"Input shape: {sample_image.shape}")
print(f"Output shape: {augmented['image'].shape}")
print(f"Output type: {type(augmented['image'])}")

# Use pre-trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()

    print(f'Epoch {epoch}: Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {correct/len(val_dataset):.4f}')
