import cv2
import numpy as np

def replicate_last_frame(video_path, output_path, duration_to_replicate=10):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the last frame
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    last_frame = frames[-1]

    # Extend the frames list by replicating the last frame
    for _ in range(int(duration_to_replicate * fps)):
        frames.append(last_frame)

    # Write the extended frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()

if __name__ == "__main__":
    input_video_path = "/Users/bhuvanrj/Desktop/cricket/ball/output_vids/output_1.mp4"
    output_video_path = "/Users/bhuvanrj/Desktop/cricket/ball/output_vids/output_1_10s.mp4"

    replicate_last_frame(input_video_path, output_video_path)