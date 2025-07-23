from PIL import Image
import numpy as np
from glob import glob

def calculate_global_channel_stats(image_paths, resize=(256, 256)):
    all_pixels = []

    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(resize)
        image = np.asarray(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        all_pixels.append(image)

    all_pixels = np.stack(all_pixels, axis=0)  # Shape: (N, H, W, C)
    all_pixels = all_pixels.reshape(-1, 3)      # Flatten to (N_pixels, 3)

    mean = all_pixels.mean(axis=0)
    std = all_pixels.std(axis=0)

    return mean.tolist(), std.tolist()

# Ambil semua path gambar
image_paths = glob('dataset/kaggleDataset/*/*')
channel_mean, channel_std = calculate_global_channel_stats(image_paths)

print("Global Channel Mean:", channel_mean)
print("Global Channel Std :", channel_std)
