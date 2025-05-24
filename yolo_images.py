from ultralytics import YOLO
model = YOLO("yolo11s-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/kaggle/input/posture-keypoints-detection/data.yaml", epochs=200, imgsz=640, batch=32)

