import os
import shutil
import random
from tqdm import tqdm

# ===== CONFIGURATION =====
SOURCE_DIR = 'dataset/kaggleDataset'  # original dataset with class folders
DEST_DIR = 'split_dataset'  # destination folder to save train/ and test/
SPLIT_RATIO = 0.8  # 80% train, 20% test
SEED = 42

random.seed(SEED)

# ===== FUNCTION TO SPLIT DATASET =====
def split_and_save_dataset():
    class_names = os.listdir(SOURCE_DIR)
    class_names = [cls for cls in class_names if os.path.isdir(os.path.join(SOURCE_DIR, cls))]

    for cls in tqdm(class_names, desc="Splitting classes"):
        cls_source_path = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(cls_source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Paths to save
        train_cls_path = os.path.join(DEST_DIR, 'train', cls)
        test_cls_path = os.path.join(DEST_DIR, 'test', cls)
        os.makedirs(train_cls_path, exist_ok=True)
        os.makedirs(test_cls_path, exist_ok=True)

        # Copy files
        for img in train_imgs:
            src = os.path.join(cls_source_path, img)
            dst = os.path.join(train_cls_path, img)
            shutil.copyfile(src, dst)

        for img in test_imgs:
            src = os.path.join(cls_source_path, img)
            dst = os.path.join(test_cls_path, img)
            shutil.copyfile(src, dst)

    print("✅ Dataset successfully split into train/test folders.")

# ===== EXECUTE =====
if __name__ == '__main__':
    split_and_save_dataset()
