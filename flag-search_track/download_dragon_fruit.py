import os
import xml.etree.ElementTree as ET

# Paths
xml_dir = 'E:/electron2025/flag-search_track/fruit-detection/annotations'
output_dir = 'E:/electron2025/flag-search_track/fruit-detection/labels/train'
os.makedirs(output_dir, exist_ok=True)

# Define your class mapping
class_map = {
    "pineapple": 0,
    "snake fruit": 1,
    "dragon fruit": 2
}

# Loop through all XML files
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in class_map:
            continue

        class_id = class_map[label]

        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Save YOLO annotation with same filename but .txt
    base_filename = os.path.splitext(xml_file)[0]
    with open(os.path.join(output_dir, f"{base_filename}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))
