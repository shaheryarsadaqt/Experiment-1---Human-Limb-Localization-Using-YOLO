from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model with your dataset
model.train(
    data='D:/AI Project/datasets/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='D:/AI Project/runs',
    name='limb_detection'
)
