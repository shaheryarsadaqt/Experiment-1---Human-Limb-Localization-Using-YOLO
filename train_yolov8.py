from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or your custom model
model.train(
    data="D:\\AI Project\\data.yaml",
    epochs=50,
    imgsz=640,
    device=0,  # ← Set to GPU (0) if available
    workers=4,  # ← Enable parallel loading (adjust based on CPU cores)
    batch=16,   # ← Increase if GPU memory allows
    optimizer="AdamW"  # ← Often better than SGD
)