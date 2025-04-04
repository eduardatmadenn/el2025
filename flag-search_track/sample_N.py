import os
import random
import shutil

# Source folder with 5000 images
source_folder = 'E:/electron2025/flag-search_track/coco-2017/validation/data'

# Destination folder for the 1000 sampled images
dest_folder = 'E:/electron2025/flag-search_track/coco-2017/sampled_1000'
os.makedirs(dest_folder, exist_ok=True)

# Get all JPG filenames
all_images = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# Randomly choose 1000 images
sampled_images = random.sample(all_images, 1000)

# Copy and rename
for idx, filename in enumerate(sampled_images, 1):
    src_path = os.path.join(source_folder, filename)
    new_name = f"img_{idx:04d}.jpg"  # e.g., img_0001.jpg
    dst_path = os.path.join(dest_folder, new_name)
    shutil.copy(src_path, dst_path)

print("✅ Done! 1000 images copied and renamed.")
