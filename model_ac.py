from ultralytics import YOLO
import cv2
import math

# Initialize YOLOv8 model
model = YOLO('/Users/bhuvanrj/Desktop/cricket/ball/last_100.pt')


results = model('/Users/bhuvanrj/Desktop/1minvid.mp4', save=True)
