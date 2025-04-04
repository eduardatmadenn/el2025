from ultralytics import YOLO
import os
from PIL import Image
from tqdm import tqdm
from collections import Counter

# Load pre-trained YOLOv8 model (trained on COCO)
model = YOLO("yolov8s.pt")  # or yolov8n.pt for even faster

# Path to your images folder
image_folder = 'E:/electron2025/flag-search_track/coco-2017/validation/data'

# Collect results
results_dict = {}

# Run inference
for filename in tqdm(os.listdir(image_folder)):
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(image_folder, filename)
    results = model(image_path)[0]  # Run inference, take first result

    # Get class names
    labels = set([model.names[int(cls)] for cls in results.boxes.cls])
    results_dict[filename] = list(labels)

all_labels = [label for labels in results_dict.values() for label in labels]

# Count occurrences
class_counts = Counter(all_labels)

# Print sorted by frequency
for cls, count in class_counts.most_common():
    print(f"{cls}: {count}")

# Find all images that contain 'banana'
banana_images = [filename for filename, labels in results_dict.items() if 'banana' in labels]

print(f"\n🍌 Images containing bananas ({len(banana_images)} total):")
for fname in banana_images:
    print(fname)

with open('banana_images.txt', 'w') as f:
    for fname in banana_images:
        f.write(fname + '\n')
