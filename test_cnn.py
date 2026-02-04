import os
import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import PatchCNN, PATCH_SIZE, STRIDE

# ===== Paths =====
TEST_DIR   = "dataset/tests/"     # can be images/ OR tests/
MASK_DIR   = "dataset/masks/"
OUTPUT_DIR = "dataset/output/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():

    model = PatchCNN().to(DEVICE)
    model.load_state_dict(torch.load("patch_cnn_model.pth", map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    print("\nTesting...")

    for fname in os.listdir(TEST_DIR):

        if not fname.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(TEST_DIR, fname)
        img = cv2.imread(img_path)

        h, w = img.shape[:2]
        pred_mask = np.zeros((h, w), dtype=np.uint8)

        # ===== Sliding patches =====
        for y in range(0, h-PATCH_SIZE+1, STRIDE):
            for x in range(0, w-PATCH_SIZE+1, STRIDE):

                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch = (patch/255.0).astype(np.float32)
                patch = np.transpose(patch, (2,0,1))

                tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred = model(tensor).argmax(1).item()

                # small circle marking
                c = PATCH_SIZE // 2
                if pred == 1:
                    cv2.circle(pred_mask, (x+c, y+c), 2, 1, -1)

        # ===== Save overlay =====
        overlay = img.copy()
        overlay[pred_mask == 1] = [0,0,255]
        result = cv2.addWeighted(img, 0.7, overlay, 0.4, 0)

        out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_cnn.png"))
        cv2.imwrite(out_path, result)
        print(fname, "-> saved")

        # =================================================
        # SAFE METRICS (only if mask exists)
        # =================================================
        mask_name = fname.replace(".jpg", ".png")
        mask_path = os.path.join(MASK_DIR, mask_name)

        if os.path.exists(mask_path):   # ⭐ SAFE CHECK
            gt = cv2.imread(mask_path)
            gt_bin = (gt[:,:,2] > 150).astype(np.uint8)

            y_true.extend(gt_bin.flatten())
            y_pred.extend(pred_mask.flatten())
        else:
            print(f"⚠ No mask for {fname} → skipping accuracy")


    # ===== Evaluation =====
    if len(y_true) > 0:
        print("\n📊 Evaluation")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    else:
        print("\n(No masks found → Accuracy skipped)")


if __name__ == "__main__":
    test()
