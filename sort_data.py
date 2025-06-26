import os
import shutil
import pandas as pd

# Reload the CSV just in case
csv_path = "data/Training_Set/RFMiD_Training_Labels.csv"

df = pd.read_csv(csv_path)

# Paths setup (you can modify these)
source_image_dir = 'data/Training_Set/Training'  # Folder where all images are located (e.g., 1.jpg, 2.jpg)
output_dir = 'data'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# Create output folders
for split in [train_dir, val_dir]:
    os.makedirs(os.path.join(split, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(split, 'diseased'), exist_ok=True)

# Shuffle and split (80% train, 20% val)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx]
val_df = df[split_idx:]

# Function to copy images
def copy_images(dataframe, split_dir):
    for _, row in dataframe.iterrows():
        image_filename = f"{row['ID']}.png"
        label = 'diseased' if row['Disease_Risk'] == 1 else 'normal'
        src = os.path.join(source_image_dir, image_filename)
        dst = os.path.join(split_dir, label, image_filename)
        if os.path.exists(src):  # Only copy if image file exists
            shutil.copyfile(src, dst)

# Copy images
copy_images(train_df, train_dir)
copy_images(val_df, val_dir)

output_dir