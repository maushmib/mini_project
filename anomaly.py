import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
from math import ceil
import string
from model import PatchCNN, PATCH_SIZE, STRIDE


# =====================================================
# USER SETTINGS
# =====================================================
video_path = "wp1.mp4"
gps_log_path = "gps_log.csv"

output_frames_dir = "dataset/tests"
output_overlay_dir = "dataset/output"

target_fps = 1
cell_size_m = 10
# =====================================================


os.makedirs(output_frames_dir, exist_ok=True)
os.makedirs(output_overlay_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# STEP 1 — Extract frames
# =====================================================
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(original_fps / target_fps))

frames = []
timestamps = []

count = 0
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        frames.append(frame)
        timestamps.append(frame_no)

        cv2.imwrite(
            os.path.join(output_frames_dir, f"frame{frame_no}.jpg"),
            frame
        )

        frame_no += 1

    count += 1

cap.release()
print(f"Extracted {len(frames)} frames.")


# =====================================================
# STEP 2 — Read GPS
# =====================================================
gps_data = []

with open(gps_log_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        gps_data.append({
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"])
        })

lats = [g["latitude"] for g in gps_data]
lons = [g["longitude"] for g in gps_data]

min_lat, max_lat = min(lats), max(lats)
min_lon, max_lon = min(lons), max(lons)

avg_lat = (min_lat + max_lat) / 2


# =====================================================
# STEP 3 — PURE NATURAL GRID (NO HARD CODING)
# =====================================================
cell_lat = cell_size_m / 111000
cell_lon = cell_size_m / (111000 * np.cos(np.radians(avg_lat)))

grid_rows = ceil((max_lat - min_lat) / cell_lat)
grid_cols = ceil((max_lon - min_lon) / cell_lon)

grid = np.zeros((grid_rows, grid_cols))

print(f"Grid created: {grid_rows} rows × {grid_cols} cols")


# =====================================================
# STEP 4 — Load CNN
# =====================================================
model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("patch_cnn_model.pth", map_location=DEVICE))
model.eval()


# =====================================================
# STEP 5 — CNN predictions
# =====================================================
anomaly_results = []

for i, frame in enumerate(frames):

    h, w = frame.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h-PATCH_SIZE+1, STRIDE):
        for x in range(0, w-PATCH_SIZE+1, STRIDE):

            patch = frame[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch = (patch/255.0).astype(np.float32)
            patch = np.transpose(patch, (2,0,1))

            tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(tensor).argmax(1).item()

            if pred == 1:
                cv2.circle(pred_mask, (x+PATCH_SIZE//2, y+PATCH_SIZE//2), 2, 1, -1)

    overlay = frame.copy()
    overlay[pred_mask == 1] = [0,0,255]

    cv2.imwrite(
        os.path.join(output_overlay_dir, f"frame{i}_cnn.png"),
        cv2.addWeighted(frame, 0.7, overlay, 0.4, 0)
    )

    anomaly_results.append(1 if np.any(pred_mask==1) else 0)


# =====================================================
# STEP 6 — Map GPS → Grid (PURE math)
# =====================================================
for i in range(min(len(frames), len(gps_data))):

    lat = gps_data[i]["latitude"]
    lon = gps_data[i]["longitude"]

    row = int((lat - min_lat) / cell_lat)
    col = int((lon - min_lon) / cell_lon)

    row = np.clip(row, 0, grid_rows-1)
    col = np.clip(col, 0, grid_cols-1)

    if anomaly_results[i] == 1:
        grid[row, col] += 1


# =====================================================
# STEP 7 — Visualization WITH LABELS
# =====================================================
plt.figure(figsize=(8,6))

plt.imshow(grid, cmap="Reds", origin="lower")
plt.colorbar(label="Anomaly Count")

letters = list(string.ascii_uppercase)

for r in range(grid_rows):
    for c in range(grid_cols):

        label = f"{letters[c]}{r+1}\n{int(grid[r,c])}"

        plt.text(
            c, r,
            label,
            ha='center',
            va='center',
            color='black',
            fontsize=9,
            fontweight='bold'
        )

plt.title("Anomaly Grid Map (Labeled)")
plt.xlabel("Longitude Cells")
plt.ylabel("Latitude Cells")

plt.grid(True)
plt.show()
