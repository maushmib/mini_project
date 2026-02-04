import cv2
import os

video_path = "wp1.mp4"
output_folder = "dataset/tests"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

original_fps = cap.get(cv2.CAP_PROP_FPS)
print("Original FPS:", original_fps)

target_fps = 1
frame_interval = int(original_fps / target_fps)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        name = f"test{saved}.jpg"
        cv2.imwrite(os.path.join(output_folder, name), frame)
        saved += 1

    count += 1

cap.release()
print("Done! Total frames saved:", saved)
