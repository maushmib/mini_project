import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib
import time

# -----------------------------
# Settings
# -----------------------------
TEST_DIR = "dataset/tests/"
BLOCK_SIZE = 220  # each block size in grid
PATCH_SIZE = 3
STRIDE = 5
ANOMALY_THRESHOLD = 5  # % pixels
LIVE_DELAY = 0.5  # seconds delay to simulate drone scanning

# -----------------------------
# Load trained Random Forest
# -----------------------------
model = joblib.load("offtype_model.pkl")

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_features_patch(img_patch):
    color_mean = img_patch.mean(axis=(0,1)).tolist()
    gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    return color_mean + [contrast, homogeneity, energy, correlation]

# -----------------------------
# Load test images
# -----------------------------
files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))])
if len(files) == 0:
    print("No test images found!")
    exit()

print("Files found:", files)

# Automatic grid size
total_blocks = len(files)
GRID_COLS = int(np.ceil(np.sqrt(total_blocks)))
GRID_ROWS = int(np.ceil(total_blocks / GRID_COLS))

# -----------------------------
# Create initial grid (all green)
# -----------------------------
grid = np.ones((GRID_ROWS*BLOCK_SIZE, GRID_COLS*BLOCK_SIZE, 3), dtype=np.uint8) * 0
grid[:] = (0, 255, 0)  # GREEN

# -----------------------------
# Show initial grid window
# -----------------------------
cv2.namedWindow("Farm Map - Live", cv2.WINDOW_NORMAL)
cv2.imshow("Farm Map - Live", grid)
cv2.waitKey(1)

# -----------------------------
# Process each block / image
# -----------------------------
for i, fname in enumerate(files):
    img_path = os.path.join(TEST_DIR, fname)
    original_img = cv2.imread(img_path)
    if original_img is None:
        continue

    h, w = original_img.shape[:2]
    scale = 0.5
    img = cv2.resize(original_img, (int(w*scale), int(h*scale)))
    h_small, w_small = img.shape[:2]

    mask_small = np.zeros((h_small, w_small), dtype=np.uint8)

    # Patch-wise prediction
    for y in range(0, h_small-PATCH_SIZE+1, STRIDE):
        for x in range(0, w_small-PATCH_SIZE+1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            feat = extract_features_patch(patch)
            pred = model.predict([feat])[0]
            if pred == 1:
                mask_small[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255

    # Resize mask back to original size
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    anomaly_pixels = np.count_nonzero(mask)
    percent = 100.0 * anomaly_pixels / mask.size
    print(f"Block {i+1} ({fname}): anomaly {percent:.2f}%")

    # Grid coordinates
    r = i // GRID_COLS
    c = i % GRID_COLS
    y1 = r * BLOCK_SIZE
    y2 = y1 + BLOCK_SIZE
    x1 = c * BLOCK_SIZE
    x2 = x1 + BLOCK_SIZE

    # Update block color live
    color = (0, 0, 255) if percent > ANOMALY_THRESHOLD else (0, 255, 0)
    cv2.rectangle(grid, (x1, y1), (x2, y2), color, -1)

    # Block number
    cv2.putText(grid,
                f"B{i+1}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2)

    # Show live grid
    cv2.imshow("Farm Map - Live", grid)
    cv2.waitKey(1)
    time.sleep(LIVE_DELAY)  # simulate scanning

# -----------------------------
# Keep final grid open and save
# -----------------------------
cv2.imshow("Farm Map - Live", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("farm_result_rf_live.jpg", grid)
print("Saved final result as farm_result_rf_live.jpg")
