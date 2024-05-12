from ultralytics import YOLO


model = YOLO("/Users/bhuvanrj/Desktop/last_new.pt")

model.predict("/Users/bhuvanrj/Desktop/1minvid_zoomed.mp4", show=True)

