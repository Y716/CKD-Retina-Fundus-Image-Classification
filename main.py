import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from PIL import ImageFile
from torch.utils.data import random_split

# Prevent crash from truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# CONFIGURATION
# =========================

CONFIG = {
    'model_list': ['resnet_18', 'swin_t', 'vgg16', 'densenet121', 'efficientnet_b0'],
    'batch_size': 8,
    'epochs': 10,
    'learning_rate': 1e-4,
    'img_size': 224,
    'num_workers': 2,
    'wandb_project': 'klasifikasi-fundus-retina'
}

DATASET_DIR = os.path.join('dataset')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# =========================
# DATASET & TRANSFORMS
# =========================

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}

# Load full dataset from class-labeled folders
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=data_transforms['train'])  # transform only used as placeholder

# Train-val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Update transforms (ImageFolder doesn't transform split subsets directly)
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

# Class names from ImageFolder
class_names = full_dataset.classes
print('Classes:', class_names)


# =========================
# MODEL SELECTOR
# =========================

def get_model(name, num_classes):
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == 'swin_t':
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif name == 'resnet_18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f'Unsupported model: {name}')
    return model.to(device)

# =========================
# TRAINING FUNCTION
# =========================

def train_model(model, model_name, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=f"{model_name.upper()} - {phase.capitalize()} Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            wandb.log({f'{phase}_loss': epoch_loss,
                       f'{phase}_acc': epoch_acc,
                       'epoch': epoch+1})

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{model_name}_best.pth'))
                print(f'✅ Best model saved: {model_name}_best.pth — Acc: {best_acc:.4f}')

    return best_acc

# =========================
# MAIN LOOP FOR ALL MODELS
# =========================

if __name__ == '__main__':
    for model_name in CONFIG['model_list']:
        print(f'\n🔁 Starting training for model: {model_name}')
        wandb.init(project=CONFIG['wandb_project'], name=f'{model_name}_training', config={**CONFIG, 'model_name': model_name})

        model = get_model(model_name, len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

        acc = train_model(model, model_name, criterion, optimizer, CONFIG['epochs'])

        wandb.run.summary[f'{model_name}_val_acc'] = acc
        wandb.finish()

    print("✅ All models trained!")