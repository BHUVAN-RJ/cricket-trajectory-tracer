from ultralytics import YOLO
import cv2
import math
import time
import numpy as np
from sahi.sahi.auto_model import AutoDetectionModel
from sahi.sahi.predict import get_sliced_prediction

coordinates_of_quad = [[942, 324], [1193, 339], [254, 456], [13, 416]]
video_path = '/Users/bhuvanrj/Desktop/cricket_sahi_clips/clip1.mp4'
model_path = '/Users/bhuvanrj/Desktop/cricket/ball/last_100.pt'
frame_1 = None
frame_with_text = None
current_centers = []
all_centers = []
max_y_value = -1
cap = cv2.VideoCapture(video_path)
start_time = 0
end_time = 0
distance = 22 * 0.9144

output_video_path = '/Users/bhuvanrj/Desktop/cricket/ball/output_vids/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.55,
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
    #         # Calculate the center of the bounding box
    #         center = (int(x + width / 2), int(y + height / 2))
    #
    #         current_centers.append(center)
    #
    #         # Draw rectangle on the image
    #         cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)
    #
    # # Draw red circles after drawing rectangles
    # for center in current_centers:
    #     cv2.circle(frame, center, 5, (0, 0, 255), -1)
    #
    # # Connect the dots with lines
    # for i in range(len(current_centers) - 1):
    #     if len(current_centers) == 6:
    #         end_time = time.time()
    #         print(end_time)
    #
    #     if len(current_centers) >= 6:
    #         break
    #     cv2.line(frame, current_centers[i], current_centers[i + 1], (255, 0, 0), 2)
    #     cv2.imwrite(f'/Users/bhuvanrj/Desktop/cricket/ball/output_vids/frames/frame{frames_count}.jpg', frame)
    #
    # # Calculate speed
    # if start_time != 0 and end_time != 0:
    #     time_taken = end_time - start_time
    #     speed_yard_per_sec = distance / time_taken
    #     speed_kmph = speed_yard_per_sec * 0.000568182 * 3600  # Convert to kilometers per hour
    #
    #     # Add speed to the final frame as text
    #     cv2.putText(frame, f"Speed: 47.039 km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the image with bounding boxes
    out.write(frame)
    final_frame = frame
    frame_with_text = frame.copy()

# Draw quadrilateral using the coordinates_of_quad
cv2.polylines(final_frame, [np.array(coordinates_of_quad)], isClosed=True, color=(0, 255, 0), thickness=2)

# Draw divisions on the length part of the lines
divisions = [2, 6, 7, 8]  # Divisions at 2m, 6m, 7m, 8m on the length part of the lines
ax, ay  = coordinates_of_quad[0][0], coordinates_of_quad[0][1]
bx, by  = coordinates_of_quad[1][0], coordinates_of_quad[1][1]
cx, cy  = coordinates_of_quad[2][0], coordinates_of_quad[2][1]
dx, dy  = coordinates_of_quad[3][0], coordinates_of_quad[3][1]

dist_ad = math.sqrt(math.pow((dx-ax),2) + math.pow((dy-ay),2))
dist_bc = math.sqrt(math.pow((cx-bx),2) + math.pow((cy-by),2))
div_yorker = dist_ad * (2/20.12)
div_full = dist_ad * (6/20.12)
div_good = dist_ad * (7/20.12)
div_back = dist_ad * (8/20.12)


div_yorker = dist_bc * (2/20.12)
div_full = dist_bc * (6/20.12)
div_good = dist_bc * (7/20.12)
div_back = dist_bc * (8/20.12)
cv2.circle(final_frame, (107,406), 5, (0, 0, 255), -1)
cv2.circle(final_frame, (340,383), 5, (0, 0, 255), -1)
cv2.circle(final_frame, (294,388), 5, (0, 0, 255), -1)
cv2.circle(final_frame, (387,378), 5, (0, 0, 255), -1)
cv2.circle(final_frame, (627,411), 5, (0, 0, 255), -1)
cv2.circle(final_frame, (581,417), 5, (0, 0, 255), -1)
cv2.circle(final_frame, (534,422), 5, (0, 0, 255), -1)
cv2.circle(final_frame, (347,445), 5, (0, 0, 255), -1)
cv2.line(final_frame, (107,406), (347,445), (255, 0, 0), 2)
cv2.line(final_frame, (340,383),(581,417) , (255, 0, 0), 2)
cv2.line(final_frame, (294,388), (534,422), (255, 0, 0), 2)
cv2.line(final_frame, (387,378), (627,411), (255, 0, 0), 2)
cv2.imwrite('/Users/bhuvanrj/Desktop/cricket/ball/output_vids/final_frame.jpg', final_frame)

# Release video capture and writer
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
