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
# CNN Model (same as before)
# ---------------------------
class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*PATCH_SIZE*PATCH_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("cnn_model.pth", map_location=DEVICE))
model.eval()

# ---------------------------
# Accuracy counters
# ---------------------------
y_true = []
y_pred = []

# ---------------------------
# Process test images
# ---------------------------
for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(TEST_DIR, fname)
    original_img = cv2.imread(img_path)
    if original_img is None:
        continue

    h, w = original_img.shape[:2]

    # Resize for faster processing
    scale = 0.5
    img = cv2.resize(original_img, (int(w*scale), int(h*scale)))
    h_small, w_small = img.shape[:2]

    # Prepare prediction mask
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

    # Resize mask back to original size
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask > 0).astype(np.uint8)

    # ---------------------------
    # Apply red mask on original image
    # ---------------------------
    result = original_img.copy()
    result[mask_binary==1] = [0,0,255]

    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_output.png"))
    cv2.imwrite(out_path, result)
    print("Saved:", out_path)

    # ---------------------------
    # Load ground-truth mask
    # ---------------------------
    # map test1.jpg -> img1.png
    num = fname.replace("test","").replace(".jpg","")
    mask_path = os.path.join(MASK_DIR, f"img{num}.png")
    gt = cv2.imread(mask_path)
    if gt is None:
        print("Skipping ground truth for:", fname)
        continue

    gt = cv2.resize(gt, (w, h))
    gt_binary = (gt[:,:,2] > 150).astype(np.uint8)  # same logic as train.py / test.py

    # Flatten for pixel-level evaluation
    y_true.extend(gt_binary.flatten())
    y_pred.extend(mask_binary.flatten())

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

    # IoU (Intersection over Union) for Off-type
    intersection = np.logical_and(y_true==1, y_pred==1).sum()
    union = np.logical_or(y_true==1, y_pred==1).sum()
    iou = intersection / union if union > 0 else 0
    print(f"Pixel-level IoU for Off-type: {iou:.4f}")

else:
    print("⚠ No ground truth masks found or all skipped")
