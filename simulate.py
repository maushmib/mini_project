import os
import cv2
import numpy as np
import torch
from model import PatchCNN, PATCH_SIZE, STRIDE

# ===============================
# SETTINGS
# ===============================
TEST_DIR = "dataset/tests/"
OUT_DIR  = "dataset/output/"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CELL = 80   # size of each grid block (pixels)

# ===== Initial GPS (fake simulation) =====
START_LAT = 11.0168
START_LON = 76.9558
STEP = 0.00001   # movement step


# ===============================
# Load Model
# ===============================
model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("patch_cnn_model.pth", map_location=DEVICE))
model.eval()


# ===============================
# Get images
# ===============================
files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")])
N = len(files)

# Auto grid shape (square-ish)
cols = int(np.ceil(np.sqrt(N)))
rows = int(np.ceil(N / cols))


# ===============================
# Create grid (brown)
# ===============================
grid_img = np.zeros((rows*CELL, cols*CELL, 3), dtype=np.uint8)
grid_img[:] = (42, 42, 165)  # brown


def has_anomaly(img):
    """Check whole image using sliding patches"""
    h, w = img.shape[:2]

    for y in range(0, h-PATCH_SIZE+1, STRIDE):
        for x in range(0, w-PATCH_SIZE+1, STRIDE):

            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch = (patch/255.0).astype(np.float32)
            patch = np.transpose(patch, (2,0,1))

            t = torch.tensor(patch).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(t).argmax(1).item()

            if pred == 1:
                return True

    return False


# ===============================
# Zigzag path generation
# ===============================
path = []
for r in range(rows):
    if r % 2 == 0:
        for c in range(cols):
            path.append((r, c))
    else:
        for c in reversed(range(cols)):
            path.append((r, c))


# ===============================
# Simulation
# ===============================
lat = START_LAT
lon = START_LON

print("\n🚁 Drone Simulation Started\n")

for idx, fname in enumerate(files):

    r, c = path[idx]

    img = cv2.imread(os.path.join(TEST_DIR, fname))

    anomaly = has_anomaly(img)

    # ===== Mark grid color =====
    y1, y2 = r*CELL, (r+1)*CELL
    x1, x2 = c*CELL, (c+1)*CELL

    if anomaly:
        grid_img[y1:y2, x1:x2] = (0,0,0)      # black
        status = "ANOMALY"
    else:
        grid_img[y1:y2, x1:x2] = (42,42,165)  # brown
        status = "NORMAL"

    # ===== Draw border =====
    cv2.rectangle(grid_img, (x1,y1), (x2,y2), (255,255,255), 2)

    # ===== Label (a1, b3...) =====
    label = f"{chr(97+r)}{c+1}"
    cv2.putText(grid_img, label, (x1+10,y1+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # ===== Simulated GPS move =====
    lon += STEP

    print(f"{label} | {status} | lat={lat:.5f} lon={lon:.5f}")

# ===============================
# Save final grid map
# ===============================
cv2.imwrite(os.path.join(OUT_DIR, "grid_map.png"), grid_img)

print("\n✅ Grid map saved -> dataset/output/grid_map.png")
