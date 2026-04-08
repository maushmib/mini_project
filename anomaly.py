import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import datetime
from math import ceil
import string
import sys
from model import PatchCNN, PATCH_SIZE, STRIDE


video_path       = sys.argv[1]
gps_log_path     = sys.argv[2]
output_grid_path = sys.argv[3]
cmd_bounds_arg   = sys.argv[4] if len(sys.argv) > 4 else ""

# ── Folder structure ──────────────────────────────────────────────────────────
FRAMES_DIR  = "frames"          # ALL extracted frames
OUTPUT_DIR  = "output"          # ONLY anomaly frames
# Keep legacy dirs so existing code that reads them still works
LEGACY_FRAMES_DIR  = "dataset/tests"
LEGACY_OVERLAY_DIR = "dataset/output"

for d in (FRAMES_DIR, OUTPUT_DIR, LEGACY_FRAMES_DIR, LEGACY_OVERLAY_DIR):
    os.makedirs(d, exist_ok=True)

cell_size_m = 10
HYBRID_RATIO_THRESHOLD = 0.01   # lowered from 0.03 — fires if >1% of patches are anomaly
CONF_THRESHOLD         = 0.55   # fallback: flag frame if any patch confidence exceeds this

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Frame Extraction
# =========================

cap = cv2.VideoCapture(video_path)
original_fps   = cap.get(cv2.CAP_PROP_FPS)
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS            : {original_fps:.2f}")
print(f"Total frames in video: {total_video_frames}")

frames     = []
timestamps = []
count      = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save EVERY frame to frames/
    cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{count}.jpg"), frame)

    # Also save at 1 FPS to legacy dir for backward compatibility
    frame_interval = max(1, int(original_fps / 1))
    if count % frame_interval == 0:
        frames.append(frame)
        timestamps.append(count / original_fps)
        cv2.imwrite(os.path.join(LEGACY_FRAMES_DIR, f"frame{len(frames)-1}.jpg"), frame)

    count += 1

cap.release()

print(f"Total frames extracted : {count}")
print(f"Frames used for detection (1 FPS): {len(frames)}")


# =========================
# Read GPS Data
# =========================

gps_data = []

with open(gps_log_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S.%f")
        gps_data.append({
            "time": t,
            "lat":  float(row["lat"]),
            "lon":  float(row["lon"])
        })

lats = [g["lat"] for g in gps_data]
lons = [g["lon"] for g in gps_data]

min_lat, max_lat = min(lats), max(lats)
min_lon, max_lon = min(lons), max(lons)
avg_lat = (min_lat + max_lat) / 2

# Use CMD waypoint bounds for grid if provided — covers the real field area.
# Fall back to GPS path bounds if not available.
if cmd_bounds_arg:
    parts = [float(v) for v in cmd_bounds_arg.split(",")]
    grid_min_lat, grid_max_lat = parts[0], parts[1]
    grid_min_lon, grid_max_lon = parts[2], parts[3]
    print(f"Grid bounds : CMD waypoints ({grid_max_lat - grid_min_lat:.6f}° lat "
          f"x {grid_max_lon - grid_min_lon:.6f}° lon)")
else:
    grid_min_lat, grid_max_lat = min_lat, max_lat
    grid_min_lon, grid_max_lon = min_lon, max_lon
    print("Grid bounds : GPS path (CMD bounds not provided)")

avg_grid_lat = (grid_min_lat + grid_max_lat) / 2

cell_lat = cell_size_m / 111000
cell_lon = cell_size_m / (111000 * np.cos(np.radians(avg_grid_lat)))

grid_rows = max(1, ceil((grid_max_lat - grid_min_lat) / cell_lat))
grid_cols = max(1, ceil((grid_max_lon - grid_min_lon) / cell_lon))
grid      = np.zeros((grid_rows, grid_cols))

print(f"Grid created: {grid_rows} rows x {grid_cols} cols")


# =========================
# Load CNN Model
# =========================

model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("patch_cnn_model.pth", map_location=DEVICE))
model.eval()
print("Model loaded successfully")


# =========================
# Detect Hybrid Plants
# =========================

anomaly_results  = []
total_anomalies  = 0

BATCH_SIZE = 512    # patches per forward pass

for i, frame in enumerate(frames):

    h, w = frame.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    # Collect all patches and their positions
    patches, positions = [], []
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = frame[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch = (patch / 255.0).astype(np.float32)
            patches.append(np.transpose(patch, (2, 0, 1)))
            positions.append((x, y))

    # Run model in batches
    hybrid_patch_count = 0
    max_confidence     = 0.0
    all_preds, all_confs = [], []

    for b in range(0, len(patches), BATCH_SIZE):
        batch_tensor = torch.tensor(np.stack(patches[b:b+BATCH_SIZE])).to(DEVICE)
        with torch.no_grad():
            logits = model(batch_tensor)
            preds  = logits.argmax(1).cpu().numpy()
            confs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(preds)
        all_confs.extend(confs)

    for (x, y), pred, conf in zip(positions, all_preds, all_confs):
        if conf > max_confidence:
            max_confidence = conf
        if pred == 1:
            hybrid_patch_count += 1
            cv2.circle(pred_mask, (x + PATCH_SIZE // 2, y + PATCH_SIZE // 2), 2, 1, -1)

    total_patches = len(patches)
    hybrid_ratio  = hybrid_patch_count / total_patches

    print(f"Frame {i:>4} | hybrid patches: {hybrid_patch_count:>5} | "
          f"ratio: {hybrid_ratio:.4f} | max_conf: {max_confidence:.4f}")

    is_anomaly = hybrid_ratio > HYBRID_RATIO_THRESHOLD or max_confidence > CONF_THRESHOLD
    anomaly_results.append(1 if is_anomaly else 0)

    # ── Overlay (legacy output dir) ───────────────────────────────────────────
    overlay = frame.copy()
    overlay[pred_mask == 1] = [0, 0, 255]
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.4, 0)
    cv2.imwrite(os.path.join(LEGACY_OVERLAY_DIR, f"frame{i}_cnn.png"), blended)

    # ── Anomaly frame → output/ ───────────────────────────────────────────────
    if is_anomaly:
        total_anomalies += 1
        annotated = blended.copy()
        cv2.putText(annotated, "Anomaly", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.rectangle(annotated, (5, 5), (w - 5, h - 5), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"anomaly_frame_{i}.jpg"), annotated)

    # ── Debug: print logits if no anomaly detected yet ────────────────────────
    if not is_anomaly and total_anomalies == 0 and i < 5:
        print(f"  [DEBUG] Frame {i} — ratio {hybrid_ratio:.4f} (threshold {HYBRID_RATIO_THRESHOLD}) "
              f"| max_conf {max_confidence:.4f} (threshold {CONF_THRESHOLD})")


# =========================
# Summary Log
# =========================

print()
print("=" * 45)
print(f"FPS                    : {original_fps:.2f}")
print(f"Total frames extracted : {count}")
print(f"Total frames processed : {len(frames)}")
print(f"Total anomalies detected: {total_anomalies}")
print("=" * 45)

if total_anomalies == 0:
    print("[WARNING] No anomalies detected.")
    print(f"  Ratio threshold : {HYBRID_RATIO_THRESHOLD}")
    print(f"  Conf  threshold : {CONF_THRESHOLD}")
    print("  Check the DEBUG lines above — if max_conf is always near 0.5 the model")
    print("  weights may not match the architecture. Re-train with python patch.py")


# =========================
# Map Frame → GPS Coordinate
# =========================

gps_start = gps_data[0]["time"]
gps_times = np.array([(g["time"] - gps_start).total_seconds() for g in gps_data])
gps_lats  = np.array([g["lat"] for g in gps_data])
gps_lons  = np.array([g["lon"] for g in gps_data])

# Check if GPS path is too small (drone hovered) — spread < 5m total
gps_lat_spread = (max(gps_lats) - min(gps_lats)) * 111000
gps_lon_spread = (max(gps_lons) - min(gps_lons)) * 111000 * np.cos(np.radians(np.mean(gps_lats)))
gps_spread_m   = max(gps_lat_spread, gps_lon_spread)

if gps_spread_m < 5.0:
    print(f"[WARN] GPS spread is only {gps_spread_m:.2f}m — drone hovered during video.")
    print("[WARN] Distributing detections evenly across GPS path for visualization.")
    use_distributed = True
else:
    use_distributed = False

for i, frame_time in enumerate(timestamps):

    if use_distributed:
        # Spread frames evenly across the full GPS path by index
        idx = int(i / max(len(timestamps) - 1, 1) * (len(gps_data) - 1))
        lat = gps_lats[idx]
        lon = gps_lons[idx]
    else:
        t   = float(np.clip(frame_time, gps_times[0], gps_times[-1]))
        lat = float(np.interp(t, gps_times, gps_lats))
        lon = float(np.interp(t, gps_times, gps_lons))

    row = int((lat - grid_min_lat) / cell_lat)
    col = int((lon - grid_min_lon) / cell_lon)
    row = np.clip(row, 0, grid_rows - 1)
    col = np.clip(col, 0, grid_cols - 1)

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

plt.figure(figsize=(8, 6))
plt.imshow(grid, cmap="Reds", origin="lower")
plt.colorbar(label="Anomaly Count")

letters = list(string.ascii_uppercase)

for r in range(grid_rows):
    for c in range(grid_cols):
        label = f"{letters[c]}{r+1}\n{int(grid[r,c])}"
        plt.text(c, r, label, ha="center", va="center",
                 color="black", fontsize=9, fontweight="bold")

plt.title("Anomaly Grid Map (Labeled)")
plt.xlabel("Longitude Cells")
plt.ylabel("Latitude Cells")
plt.grid(True)
plt.savefig(output_grid_path)
plt.close()

print("Grid image saved successfully!")
