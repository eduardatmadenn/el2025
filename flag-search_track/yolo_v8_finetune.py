from ultralytics import YOLO
import os

def simple_train():

    # Optional: Set TensorBoard log dir
     # Load the model
    model = YOLO("yolov8s.pt")  # You can use yolov8n.pt, yolov8m.pt, etc.

    # Train the model
    model.train(
        data="E:/electron2025/flag-search_track/fruit-detection/data.yml",
        epochs=50,
        imgsz=640,
        batch=8,  # adjust based on your VRAM
        project="fruit_yolo_train",
        name="yolov8s_fruits",
        verbose=True,
        device=0,  # or 'cpu'
        val=True
    )

if __name__ == '__main__':
    simple_train()