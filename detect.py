from ultralytics import YOLO

model = YOLO("model/yolo.pt")
results = model.track(source="test/lalu-lintas-27260.mp4",conf=0.30 ,show=True)
