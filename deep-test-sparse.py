from ultralytics import YOLO

# Load a model
model = YOLO('model/yolo-version-3.pt')  # load an official model

# Export the model
model.export(format='tflite')