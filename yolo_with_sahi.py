from ultralytics import YOLO
import cv2
import csv
import numpy as np
import time
from sahi.sahi.auto_model import AutoDetectionModel
from sahi.sahi.predict import get_sliced_prediction

coordinates_of_quad = [[942, 324], [1193, 339], [254, 456], [13, 416]]
video_path = '/Users/bhuvanrj/Desktop/cricket_sahi_clips/clip5.mp4'
model_path = ''
frame_1 = None
frame_with_text = None
current_centers = []
all_centers = []
max_y_value = -1
cap = cv2.VideoCapture(video_path)
start_time = 0
end_time = 0
distance = 22 * 0.9144
frame_count = 0
output_video_path = '/Users/bhuvanrj/Desktop/cricket/ball/output_vids/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.6,
    device="cpu",
)

cap = cv2.VideoCapture(video_path)
frame_time = []
list_of_xy = []
swing = ""

max_y_center = None
max_y_frame = None
max_y_value = -1
loop = 0

previous_detections = []
start_time = 0
frames_count = 0

while cap.isOpened():
    frame_count += 1
    success, frame = cap.read()
    if loop == 0:
        frame_1 = frame
    if not success:
        break

    frames_count += 1
    # results = get_sliced_prediction(
    #     frame,
    #     detection_model,
    #     slice_height=256,
    #     slice_width=256,
    #     overlap_height_ratio=0.2,
    #     overlap_width_ratio=0.2
    # )
    #
    # for det in results.object_prediction_list:
    #     category_name = det.category.name
    #
    #     if category_name == 'ball':
    #         start_time = time.time()
    #         bbox = det.bbox
    #
    #         x, y, width, height = bbox.to_xywh()
    #
    #         center = (int(x + width / 2), int(y + height / 2))
    #
    #         current_centers.append(center)
    #
    #         cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)
    #
    # for center in current_centers:
    #     cv2.circle(frame, center, 5, (0, 0, 255), -1)
    #
    # for i in range(len(current_centers) - 1):
    #     print(len(current_centers))
    #     if len(current_centers) == 3:
    #         end_time = time.time()
    #         print(end_time)
    #
    #     if len(current_centers) >= 3:
    #         break
    #     cv2.line(frame, current_centers[i], current_centers[i + 1], (255, 0, 0), 2)
    #
    # if start_time != 0 and end_time != 0:
    #     time_taken = end_time - start_time
    #     speed_yard_per_sec = distance / time_taken
    #     speed_kmph = speed_yard_per_sec * 0.000568182 * 3600
    #
    #     # cv2.putText(frame, f"Speed: 53.2 km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #
    # out.write(frame)
    # final_frame = frame
    # frame_with_text = frame.copy()
#
# for i in range(0,3):
#         cv2.line(final_frame, current_centers[i], current_centers[i + 1], (255, 0, 0), 2)
#
# cv2.polylines(final_frame, [np.array(coordinates_of_quad)], isClosed=True, color=(0, 255, 0), thickness=2)
# cv2.circle(final_frame, (107,406), 5, (0, 0, 255), -1)
# cv2.circle(final_frame, (340,383), 5, (0, 0, 255), -1)
# cv2.circle(final_frame, (294,388), 5, (0, 0, 255), -1)
# cv2.circle(final_frame, (387,378), 5, (0, 0, 255), -1)
# cv2.circle(final_frame, (627,411), 5, (0, 0, 255), -1)
# cv2.circle(final_frame, (581,417), 5, (0, 0, 255), -1)
# cv2.circle(final_frame, (534,422), 5, (0, 0, 255), -1)
# cv2.circle(final_frame, (347,445), 5, (0, 0, 255), -1)
# cv2.line(final_frame, (107,406), (347,445), (255, 0, 0), 2)
# cv2.line(final_frame, (340,383),(581,417) , (255, 0, 0), 2)
# cv2.line(final_frame, (294,388), (534,422), (255, 0, 0), 2)
# cv2.line(final_frame, (387,378), (627,411), (255, 0, 0), 2)
# cv2.putText(final_frame, f"Swing: Yorker", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
# for i in range(1, 100):
#     out.write(final_frame)

cap.release()
out.release()
print(frame_count)
cv2.destroyAllWindows()
