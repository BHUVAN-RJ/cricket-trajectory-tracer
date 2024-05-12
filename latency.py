from ultralytics import YOLO
import cv2
import time
import tensorflow_hub as tfhub

# Load the YOLO model
model_path = '/Users/bhuvanrj/Desktop/cricket/ball/last_100.pt'
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
start_time = time.time()
prev_time = start_time
frame_count = 0

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()


    # Resize the frame to 256x256
    # resized_frame = cv2.resize(frame, (256, 256))

    # Start timing the detection process
    curr_time = time.time()

    # Pass the resized frame through the YOLO model
    results = model(frame)[0]
    for result in results:
        boxes = result.boxes
        box_cord = boxes.xyxy[0].cpu().numpy()

        # Extract coordinates
        x1, y1, x2, y2 = map(int, box_cord[:4])

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    end_time = time.time()

    fps = 1 / (end_time - curr_time)

    # Calculate the latency and FPS
    latency = (end_time - curr_time) * 1000 # convert to milliseconds

    # Increment frame count
    frame_count += 1

    # Set previous time to current time
    prev_time = time.time()



    # Show FPS and latency on screen
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Latency: {latency:.2f} ms', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original-sized frame
    cv2.imshow('Webcam Keypoints', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()