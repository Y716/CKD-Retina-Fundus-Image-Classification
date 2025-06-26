import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# CONFIG
VAL_DIR = 'data/val'
MODEL_PATH = 'models/best_model.pth'
MISCLASSIFIED_DIR = 'misclassified'
IMG_SIZE = 224
BATCH_SIZE = 32

# Create output dir
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transform
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load validation dataset
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get class names
class_names = val_dataset.classes

# Load model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Inverse normalization for visualization
def inverse_normalize(tensor):
    return torch.clamp((tensor * 0.5) + 0.5, 0, 1)

# Find misclassified images
misclassified = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append({
                    'image': inputs[i].cpu(),
                    'pred': preds[i].item(),
                    'label': labels[i].item(),
                    'confidence': confs[i].item()
                })

# Sort by confidence (descending)
misclassified = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)

# Save misclassified images
for idx, item in enumerate(misclassified):
    img_tensor = inverse_normalize(item['image'])
    img = transforms.ToPILImage()(img_tensor)

    filename = f"img_{idx}_pred_{class_names[item['pred']]}_true_{class_names[item['label']]}_conf_{item['confidence']:.2f}.png"
    img.save(os.path.join(MISCLASSIFIED_DIR, filename))

print(f"Saved {len(misclassified)} misclassified images to '{MISCLASSIFIED_DIR}/'")
