from ultralytics import YOLO

model = YOLO("/Users/bhuvanrj/Desktop/cricket/ball/best_100.pt")

model.export(format='tflite')