import torch
import cv2
import os
import numpy as np
from train_unet import UNet, SIZE, DEVICE

TEST_DIR = "dataset/tests/"
OUTPUT_DIR = "dataset/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 3
STRIDE = 5

model = UNet().to(DEVICE)
model.load_state_dict(torch.load("leaf_unet.pth", map_location=DEVICE))
model.eval()

for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(TEST_DIR, fname)
    original_img = cv2.imread(img_path)
    h, w = original_img.shape[:2]

    # Predict full mask (fast)
    x = cv2.resize(original_img, (SIZE, SIZE)) / 255.0
    x = torch.tensor(x).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        mask_pred = model(x)[0][0].cpu().numpy()

    # Resize mask back to original size
    mask_pred = cv2.resize(mask_pred, (w,h), interpolation=cv2.INTER_NEAREST)

    # Initialize neat patch-style mask
    mask = np.zeros((h,w), dtype=np.uint8)

    # Loop over patches (small, just for marking)
    for y in range(0, h-PATCH_SIZE+1, STRIDE):
        for x in range(0, w-PATCH_SIZE+1, STRIDE):
            patch = mask_pred[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if np.any(patch > 0.3):  # patch has any anomaly
                mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255

    # Apply red markings neatly
    result = original_img.copy()
    result[mask==255] = [0,0,255]

    # Save output
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_output.png"))
    cv2.imwrite(out_path, result)
    print("Saved:", out_path)

print("All test images processed: neat red markings like old RF!")
