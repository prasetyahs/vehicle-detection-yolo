from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("model/yolo.pt")
results = model.track(source="test/lalu-lintas-27260.mp4",conf=0.15 ,show=True)