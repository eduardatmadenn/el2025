from ultralytics import YOLO
import os
from tqdm import tqdm
from collections import defaultdict, Counter

# Load your fine-tuned model
model = YOLO("yolov8newfruits.pt")

# Image folder path
image_folder = 'E:/electron2025/flag-search_track/fruit-detection/dragon_fruit_samples'

# Results with confidences
results_dict = defaultdict(list)

# Run inference
for filename in tqdm(os.listdir(image_folder)):
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(image_folder, filename)
    results = model(image_path)[0]  # first result (single image)

    for cls, conf in zip(results.boxes.cls, results.boxes.conf):
        label = model.names[int(cls)]
        results_dict[filename].append((label, float(conf)))

# Print all class detections with confidence
all_labels = [label for predictions in results_dict.values() for (label, _) in predictions]
class_counts = Counter(all_labels)

print("\n📊 Class frequencies:")
for cls, count in class_counts.most_common():
    print(f"{cls}: {count}")

# Filter images that contain 'dragon fruit'
target_class = 'dragon fruit'
dragon_fruit_images = {
    fname: [(label, conf) for label, conf in preds if label == target_class]
    for fname, preds in results_dict.items()
    if any(label == target_class for label, _ in preds)
}

print(f"\n🍉 Images containing dragon fruit ({len(dragon_fruit_images)} total):")
for fname, matches in dragon_fruit_images.items():
    print(f"{fname}: {[f'{label} ({conf:.2f})' for label, conf in matches]}")

# Save to file
with open('dragon_fruit_images.txt', 'w') as f:
    for fname, matches in dragon_fruit_images.items():
        line = f"{fname}: {[f'{label} ({conf:.2f})' for label, conf in matches]}"
        f.write(line + "\n")
