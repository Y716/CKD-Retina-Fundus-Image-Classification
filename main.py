# full_pipeline_with_albumentations.py
# =========================================================
#  End‑to‑end pipeline:
#  1.  Albumentations‑based data augmentation
#  2.  Stratified train/val split
#  3.  Preview & save augmented samples (optional)
#  4.  Multi‑model training + early‑stopping
#  5.  Confusion‑matrix logging to WandB (figure + interactive)
# =========================================================
import matplotlib.pyplot as plt
import random
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from tqdm import tqdm
import wandb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- CONFIG ----------------
CONFIG = {
    'dataset_dir'   : 'dataset/kaggleDataset',   # root with class folders
    'preview_dir'   : 'augmented_preview',       # folder to save sample aug images
    'preview_limit' : 100,                       # max preview images to save
    'model_list'    : ['mobilenet_v2', 'swin_t', 'vgg16', 'densenet121', 'efficientnet_b0'],
    'batch_size'    : 8,
    'epochs'        : 10,
    'patience'      : 3,                         # early‑stopping
    'learning_rate' : 1e-4,
    'img_size'      : 224,
    'num_workers'   : 2,
    'project'       : 'klasifikasi-fundus-retina'
}
os.makedirs(CONFIG['preview_dir'], exist_ok=True)
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# ---------------- Albumentations transform ----------------
alb_transforms = {
    'train': A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ToTensorV2(),
    ]),
    'val'  : A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ToTensorV2(),
    ]),
}

# ---------------- Custom Dataset ----------------
class AlbumentationDataset(Dataset):
    def __init__(self, img_paths, labels, transform, preview=False):
        self.img_paths, self.labels = img_paths, labels
        self.transform = transform
        self.preview    = preview
        self._saved_cnt = 0

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if img is None:
            raise FileNotFoundError(f'Cannot read {self.img_paths[idx]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        tensor_img = augmented['image']

        # save preview
        if self.preview and self._saved_cnt < CONFIG['preview_limit']:
            bgr = cv2.cvtColor(tensor_img.permute(1,2,0).cpu().numpy()*0.5+0.5, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(CONFIG['preview_dir'], f'aug_{self._saved_cnt}.png'),
                        (bgr*255).astype('uint8'))
            self._saved_cnt += 1
        return tensor_img, torch.tensor(self.labels[idx], dtype=torch.long)

# ---------------- Build dataset ----------------
full = ImageFolder(CONFIG['dataset_dir'])
img_paths = [s[0] for s in full.samples]
labels    = [s[1] for s in full.samples]
class_names = full.classes

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in sss.split(img_paths, labels):
    train_paths = [img_paths[i] for i in train_idx]
    val_paths   = [img_paths[i] for i in val_idx]
    train_lbls  = [labels[i]   for i in train_idx]
    val_lbls    = [labels[i]   for i in val_idx]

train_ds = AlbumentationDataset(train_paths, train_lbls, alb_transforms['train'], preview=True)
val_ds   = AlbumentationDataset(val_paths,   val_lbls,   alb_transforms['val'],   preview=False)
train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          num_workers=CONFIG['num_workers'])
val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'])

# ---------------- Model selector ----------------
def get_model(name, num_cls):
    if name=='efficientnet_b0':
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1]=nn.Linear(m.classifier[1].in_features,num_cls)
    elif name=='densenet121':
        m=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier=nn.Linear(m.classifier.in_features,num_cls)
    elif name=='vgg16':
        m=models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        m.classifier[6]=nn.Linear(m.classifier[6].in_features,num_cls)
    elif name=='swin_t':
        m=models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        m.head=nn.Linear(m.head.in_features,num_cls)
    elif name=='mobilenet_v2':
        m=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier=nn.Sequential(nn.Dropout(),nn.Linear(m.last_channel,num_cls))
    else:
        raise ValueError(name)
    return m.to(device)

# ---------------- Train, Early‑stop, Conf‑matrix ----------------
def train_one(model, name):
    optim_ = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    crit   = nn.CrossEntropyLoss()
    best, patience=0,0
    for ep in range(CONFIG['epochs']):
        for phase in ('train','val'):
            model.train() if phase=='train' else model.eval()
            loader = train_loader if phase=='train' else val_loader
            run_loss, run_correct=0,0
            for x,y in tqdm(loader, desc=f'{name}-{phase}-E{ep+1}'):
                x,y=x.to(device),y.to(device)
                if phase=='train': 
                    optim_.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out=model(x)
                    loss=crit(out,y) 
                    preds=out.argmax(1)
                    if phase=='train': 
                        loss.backward()
                        optim_.step()
                run_loss+=loss.item()*x.size(0)
                run_correct+=(preds==y).sum().item()
            ep_loss=run_loss/len(loader.dataset)
            ep_acc =run_correct/len(loader.dataset)
            wandb.log({f'{name}/{phase}_loss':ep_loss,
                       f'{name}/{phase}_acc':ep_acc,'epoch':ep+1})
            print(f'{phase} - loss:{ep_loss:.4f} acc:{ep_acc:.4f}')
            if phase=='val':
                if ep_acc>best:
                    best=ep_acc
                    patience=0
                    torch.save(model.state_dict(),f'{MODEL_DIR}/{name}_best.pth')
                else:
                    patience+=1
        if patience>=CONFIG['patience']:
            print('Early stop')
            break
    return best

def log_conf_matrix(model,name):
    model.eval()
    all_p,all_l=[],[]
    with torch.no_grad():
        for x,y in val_loader:
            p=model(x.to(device)).argmax(1).cpu()
            all_p+=p.tolist()
            all_l+=y.tolist()
    cm = confusion_matrix(all_l, all_p)
    fig,ax=plt.subplots(figsize=(6,6))
    disp=ConfusionMatrixDisplay(cm,display_labels=class_names)
    disp.plot(ax=ax,cmap=plt.cm.Blues)
    plt.title(name)
    wandb.log({f'{name}/conf_mat_plot': wandb.Image(fig),
               f'{name}/conf_mat': wandb.plot.confusion_matrix(
                   y_true=all_l, preds=all_p, class_names=class_names)})
    plt.close(fig)

# ---------------- Main loop ----------------
for mdl in CONFIG['model_list']:
    wandb.init(project=CONFIG['project'], name=f'{mdl}_run', config=CONFIG)
    net=get_model(mdl,len(class_names))
    best_acc=train_one(net,mdl)
    log_conf_matrix(net,mdl)
    wandb.summary[f'{mdl}_best_acc']=best_acc
    wandb.finish()
print('✅ Finished all trainings')
