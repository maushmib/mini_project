import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
from math import ceil
import string
import sys
import datetime
from model import PatchCNN, PATCH_SIZE, STRIDE


video_path = sys.argv[1]
gps_log_path = sys.argv[2]
output_grid_path = sys.argv[3]


output_frames_dir = "dataset/tests"
output_overlay_dir = "dataset/output"

target_fps = 1
cell_size_m = 10

# hybrid detection threshold
HYBRID_RATIO_THRESHOLD = 0.03


os.makedirs(output_frames_dir, exist_ok=True)
os.makedirs(output_overlay_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Frame Extraction
# =========================

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

        frame_time = count / original_fps
        timestamps.append(frame_time)

        cv2.imwrite(
            os.path.join(output_frames_dir, f"frame{frame_no}.jpg"),
            frame
        )

        frame_no += 1

    count += 1

cap.release()

print(f"Extracted {len(frames)} frames.")


# =========================
# Read GPS Data
# =========================

gps_data = []

with open(gps_log_path) as f:
    reader = csv.DictReader(f)

    for row in reader:

        t = datetime.datetime.strptime(
            row["time"], "%Y-%m-%d %H:%M:%S.%f"
        )

        gps_data.append({
            "time": t,
            "lat": float(row["lat"]),
            "lon": float(row["lon"])
        })


lats = [g["lat"] for g in gps_data]
lons = [g["lon"] for g in gps_data]

min_lat, max_lat = min(lats), max(lats)
min_lon, max_lon = min(lons), max(lons)

avg_lat = (min_lat + max_lat) / 2


cell_lat = cell_size_m / 111000
cell_lon = cell_size_m / (111000 * np.cos(np.radians(avg_lat)))

grid_rows = ceil((max_lat - min_lat) / cell_lat)
grid_cols = ceil((max_lon - min_lon) / cell_lon)

grid = np.zeros((grid_rows, grid_cols))

print(f"Grid created: {grid_rows} rows × {grid_cols} cols")


# =========================
# Load CNN Model
# =========================

model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("patch_cnn_model.pth", map_location=DEVICE))
model.eval()


# =========================
# Detect Hybrid Plants
# =========================

anomaly_results = []

for i, frame in enumerate(frames):

    h, w = frame.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    hybrid_patch_count = 0

    for y in range(0, h-PATCH_SIZE+1, STRIDE):
        for x in range(0, w-PATCH_SIZE+1, STRIDE):

            patch = frame[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch = (patch/255.0).astype(np.float32)
            patch = np.transpose(patch, (2,0,1))

            tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(tensor).argmax(1).item()

            if pred == 1:
                hybrid_patch_count += 1
                cv2.circle(pred_mask, (x+PATCH_SIZE//2, y+PATCH_SIZE//2), 2, 1, -1)

    # total patches in frame
    total_patches = ((h-PATCH_SIZE)//STRIDE + 1) * ((w-PATCH_SIZE)//STRIDE + 1)

    hybrid_ratio = hybrid_patch_count / total_patches

    print(f"Frame {i} -> hybrid patches: {hybrid_patch_count}")
    print(f"Hybrid ratio: {hybrid_ratio:.4f}")

    if hybrid_ratio > HYBRID_RATIO_THRESHOLD:
        anomaly_results.append(1)
    else:
        anomaly_results.append(0)

    overlay = frame.copy()
    overlay[pred_mask == 1] = [0,0,255]

    cv2.imwrite(
        os.path.join(output_overlay_dir, f"frame{i}_cnn.png"),
        cv2.addWeighted(frame, 0.7, overlay, 0.4, 0)
    )


# =========================
# Map Frame → GPS Coordinate
# =========================

start_time = gps_data[0]["time"]

for i, frame_time in enumerate(timestamps):

    frame_timestamp = start_time + datetime.timedelta(seconds=frame_time)

    closest = min(
        gps_data,
        key=lambda g: abs((g["time"] - frame_timestamp).total_seconds())
    )

    lat = closest["lat"]
    lon = closest["lon"]

    row = int((lat - min_lat) / cell_lat)
    col = int((lon - min_lon) / cell_lon)

    row = np.clip(row, 0, grid_rows-1)
    col = np.clip(col, 0, grid_cols-1)

    if anomaly_results[i] == 1:

        grid[row, col] += 1

        print("\nHybrid plant detected at:")
        print("Latitude:", lat)
        print("Longitude:", lon)
        print("Grid Cell:", row, col)
        print()


# =========================
# Plot Grid Map
# =========================

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

plt.savefig(output_grid_path)
plt.close()

print("Grid image saved successfully!")