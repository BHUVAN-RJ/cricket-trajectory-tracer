from ultralytics import YOLO
import cv2
import math
import time
import csv
import numpy as np

# Initialize YOLOv8 model
model = YOLO('/Users/bhuvanrj/Desktop/cricket/ball/last_100.pt')
frame_1 = None
frame_with_text = None
zones = ["Yorker", "Full length", "Good Length", "Back of length", "None"]
# Open video file
video_path = '/Users/bhuvanrj/Desktop/cricket/ball/input/clip@2.mp4'
cap = cv2.VideoCapture(video_path)
frame_time = []
list_of_xy = []
swing = ""
# Define the output video path
output_video_path = '/Users/bhuvanrj/Desktop/cricket/ball/output_vids/output_1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
previous_detections = []
start_time = time.process_time()
frames_count = 0
distance = 22 * 0.9144 # Convert 22 yards to meters

max_y_center = None
max_y_frame = None
max_y_value = -1
loop = 0
current_centers = []
csv_file_path = '/Users/bhuvanrj/Desktop/cricket/ball/selected_points.csv'
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    row = next(csv_reader)
    point_1 = tuple(map(int, row['Point 1'].split(', ')))
    point_2 = tuple(map(int, row['Point 2'].split(', ')))

vector_ref = np.array([point_2[0] - point_1[0], point_2[1] - point_1[1]])

while cap.isOpened():
    success, frame = cap.read()
    if loop == 0:
        frame_1 = frame
    #code to get the points for both length and angle calculation
    # if success:
    #     cv2.imshow("Select Two Points", frame)
    #     points = []
    #
    #
    #     def click_event(event, x, y, flags, param):
    #         if event == cv2.EVENT_LBUTTONDOWN:
    #             points.append((x, y))
    #             cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    #             cv2.imshow("Select Two Points", frame)
    #
    #
    #     cv2.setMouseCallback("Select Two Points", click_event)
    #
    #     while len(points) < 2:
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == 27:  # Press 'Esc' to exit
    #             break
    #
    #     cv2.destroyAllWindows()
    #
    #     if len(points) == 2:
    #         # Write points to CSV file
    #         csv_file_path = '/Users/bhuvanrj/Desktop/cricket/ball/selected_points.csv'
    #         with open(csv_file_path, mode='w', newline='') as csv_file:
    #             csv_writer = csv.writer(csv_file)
    #             csv_writer.writerow(['Point 1', 'Point 2'])
    #             csv_writer.writerow([f'{points[0][0]}, {points[0][1]}', f'{points[1][0]}, {points[1][1]}'])
    #
    #         print(f"Points saved to {csv_file_path}")
    # else:
    #     print("Error reading the first frame.")

    if not success:
        break

    frames_count += 1
    results = model(frame, conf=0.65)


    for det in results[0].boxes.xyxy:
        loop += 1
        cur_time = time.time()
        frame_time.append(cur_time)
        x, y, w, h = det[:4].tolist()
        x, y, w, h = int(x), int(y), int(w), int(h)
        list_of_xy.append((x, y))
        center = ((x + w) // 2, (y + h) // 2)

        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        current_centers.append(center)

        # Update max_y_center if the current center has a higher Y-value
        if center[1] > max_y_value:
            if loop < 1:
                break
            max_y_center = center
            max_y_frame = frame.copy()
            max_y_value = center[1]

    for center in previous_detections:
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    for i in range(len(previous_detections)):
        if i == 0:
            continue
        else:
            prev_center = previous_detections[i - 1]
            current_center = previous_detections[i]
            distance_per_frame = math.sqrt(
                (current_center[0] - prev_center[0]) ** 2 + (current_center[1] - prev_center[1]) ** 2)

            if distance_per_frame < 300:
                cv2.line(frame, prev_center, current_center, (255, 0, 0), 2)

            if len(current_centers) > 1:
                current_x = current_centers[-1][0]
                prev_x = current_centers[-2][0] if len(current_centers) >= 2 else None

                if prev_x is not None:
                    x_change = current_x - prev_x

                    # Set a threshold for rapid x-coordinate change
                    x_change_threshold = 10

                    if abs(x_change) > x_change_threshold:
                        # Rapid change detected, compare with the x-coordinate of max_y_value frame
                        if max_y_center is not None:
                            if current_x > max_y_center[0]:
                                swing = "In swing"
                            else:
                                swing = "Out swing"
            # Blue color

        # Convert angle from radians to degrees

    # cv2.imshow("YOLOv8 Inference", frame)

    out.write(frame)
    frame_with_text = frame.copy()
    previous_detections.extend(current_centers)

vector_detected = np.array([current_centers[-1][0] - point_1[0], current_centers[-1][1] - point_1[1]])
# angle = np.arccos(np.dot(vector_ref, vector_detected) / (np.linalg.norm(vector_ref) * np.linalg.norm(vector_detected)))
# angle_degrees = np.degrees(angle)
equivalent_pixel_distance = point_2[1] - point_1[1]
zone_1_end = 2
zone_2_end = 6
zone_3_end = 7
zone_4_end = 8
# Convert zone boundaries from meters to equivalent pixel values
zone_1_end_pixel_ext = (zone_1_end / distance) * equivalent_pixel_distance
zone_2_end_pixel_ext = (zone_2_end / distance) * equivalent_pixel_distance
zone_3_end_pixel_ext = (zone_3_end / distance) * equivalent_pixel_distance
zone_4_end_pixel_ext = (zone_4_end / distance) * equivalent_pixel_distance

zone_1_end_pixel = zone_1_end_pixel_ext + point_1[1]
zone_2_end_pixel = zone_2_end_pixel_ext + point_1[1]
zone_3_end_pixel = zone_3_end_pixel_ext + point_1[1]
zone_4_end_pixel = zone_4_end_pixel_ext + point_1[1]
# Determine the zone for max_y_value
if max_y_value <= zone_1_end_pixel:
    ball_zone = 1
elif zone_1_end_pixel < max_y_value <= zone_2_end_pixel:
    ball_zone = 2
elif zone_2_end_pixel < max_y_value <= zone_3_end_pixel:
    ball_zone = 3
elif zone_3_end_pixel < max_y_value <= zone_4_end_pixel:
    ball_zone = 4
else:
    ball_zone = None

ball_zone = ball_zone-2
# Display the ball zone
zone = f"Ball Zone: {zones[ball_zone]}"

# Display the ball zone
# Display the frame with the maximum Y-value
elapsed_time = frame_time[-2] - frame_time[2]
avg_speed = distance / 1.5

avg_speed_text = f"Speed: {(avg_speed * 3.6):.2f} km/h"
swing = f"Swing:{swing}"
# cv2.line(frame_with_text, point_1, point_2, (0, 255, 0), 2)
# cv2.line(frame_with_text, point_1, current_centers[-1], (0, 0, 255), 2)
# angle_text = f"Angle: {angle_degrees:.2f} degrees"
# cv2.putText(frame_with_text, angle_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.line(frame_with_text, (0, point_1[1]), (int(cap.get(3)), point_1[1]), (255, 255, 255), 2)  # White
cv2.putText(frame_with_text, "Yorker", (10, int(zone_1_end_pixel) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green
cv2.line(frame_with_text, (0, int(zone_1_end_pixel)), (int(cap.get(3)), int(zone_1_end_pixel)), (0, 255, 0), 2)  # Green
cv2.putText(frame_with_text, "Full length", (10, int(zone_2_end_pixel) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green
cv2.line(frame_with_text, (0, int(zone_2_end_pixel)), (int(cap.get(3)), int(zone_2_end_pixel)), (0, 0, 255), 2)  # Red
cv2.putText(frame_with_text, "Good Length", (10, int(zone_3_end_pixel) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green
cv2.line(frame_with_text, (0, int(zone_3_end_pixel)), (int(cap.get(3)), int(zone_3_end_pixel)), (255, 0, 0), 2)  # Blue
cv2.putText(frame_with_text, "Back of Length", (10, int(zone_4_end_pixel) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green
cv2.line(frame_with_text, (0, int(zone_4_end_pixel)), (int(cap.get(3)), int(zone_4_end_pixel)), (255, 255, 0), 2)  # Yellow
cv2.putText(frame_with_text, "None", (10, int(zone_4_end_pixel) +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green

cv2.putText(frame_with_text, swing, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
cv2.putText(frame_with_text, avg_speed_text, (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
cv2.putText(frame_with_text, zone, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
out.write(frame_with_text)
for i in range(1,150):
    out.write(frame_with_text)

# Release the video capture object, output video, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
