import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# Settings
# ---------------------------
PATCH_SIZE = 3
STRIDE = 5

TEST_DIR = "dataset/tests/"
MASK_DIR = "dataset/masks/"
OUTPUT_DIR = "dataset/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# CNN Model (matches training)
# ---------------------------
class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 32 filters
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), # 64 filters
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*PATCH_SIZE*PATCH_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# Load trained model
model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("cnn_model.pth", map_location=DEVICE))
model.eval()

# ---------------------------
# Pixel-level metrics counters
# ---------------------------
y_true = []
y_pred = []

# ---------------------------
# Process each test image
# ---------------------------
for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(TEST_DIR, fname)
    original_img = cv2.imread(img_path)
    if original_img is None:
        continue

    h, w = original_img.shape[:2]

    # Resize for faster patch processing
    scale = 0.5
    img = cv2.resize(original_img, (int(w*scale), int(h*scale)))
    h_small, w_small = img.shape[:2]

    # Prediction mask (small)
    mask_small = np.zeros((h_small, w_small), dtype=np.uint8)

    # ---------------------------
    # Patch-level prediction
    # ---------------------------
    for y in range(0, h_small-PATCH_SIZE+1, STRIDE):
        for x in range(0, w_small-PATCH_SIZE+1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] / 255.0
            patch = np.transpose(patch, (2,0,1))
            tensor = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                pred = model(tensor).argmax(1).item()

            if pred == 1:
                mask_small[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255

    # Resize mask to original size
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask > 0).astype(np.uint8)

    # Apply red mask
    result = original_img.copy()
    result[mask_binary==1] = [0,0,255]

    # Save output
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_output.png"))
    cv2.imwrite(out_path, result)

    # Compute off-type percentage for this image
    off_percent = 100.0 * np.count_nonzero(mask_binary) / (mask_binary.shape[0] * mask_binary.shape[1])
    print(f"{fname} -> {off_percent:.2f}% off-type, saved to {out_path}")

    # ---------------------------
    # Load corresponding ground truth mask
    # map test1.jpg -> img1.png
    num = fname.replace("test","").replace(".jpg","")
    mask_path = os.path.join(MASK_DIR, f"img{num}.png")
    gt = cv2.imread(mask_path)
    if gt is not None:
        gt = cv2.resize(gt, (w, h))
        gt_binary = (gt[:,:,2] > 150).astype(np.uint8)  # same logic as training
        y_true.extend(gt_binary.flatten())
        y_pred.extend(mask_binary.flatten())
    else:
        print(f"Skipping ground truth for: {fname}")

# ---------------------------
# Pixel-level metrics
# ---------------------------
if len(y_true) > 0:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n📊 Pixel-level Evaluation:")
    print(classification_report(y_true, y_pred, target_names=["Normal","Off-type"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # IoU for Off-type
    intersection = np.logical_and(y_true==1, y_pred==1).sum()
    union = np.logical_or(y_true==1, y_pred==1).sum()
    iou = intersection / union if union > 0 else 0
    print(f"Pixel-level IoU for Off-type: {iou:.4f}")
else:
    print("⚠ No ground truth masks found or all skipped")
