import os
import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import PatchCNN, PATCH_SIZE, STRIDE

# ===== Paths =====
TEST_DIR   = "dataset/tests/"
MASK_DIR   = "dataset/masks/"
OUTPUT_DIR = "dataset/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():

    # ===== Load model =====
    model = PatchCNN().to(DEVICE)
    model.load_state_dict(torch.load("patch_cnn_model.pth", map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    print("\nTesting...")

    for fname in os.listdir(TEST_DIR):

        if not fname.lower().endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(TEST_DIR, fname))
        h, w = img.shape[:2]

        pred_mask = np.zeros((h, w), dtype=np.uint8)

        # ===== Sliding patches =====
        for y in range(0, h-PATCH_SIZE+1, STRIDE):
            for x in range(0, w-PATCH_SIZE+1, STRIDE):

                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch = (patch / 255.0).astype(np.float32)
                patch = np.transpose(patch, (2,0,1))

                tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred = model(tensor).argmax(1).item()

                # ⭐ Mark a small circle instead of a single pixel
                c = PATCH_SIZE // 2
                if pred == 1:
                    cv2.circle(pred_mask, (x+c, y+c), 2, 1, -1)  # radius=2, filled circle

        # ===== Overlay visualization =====
        overlay = img.copy()
        overlay[pred_mask==1] = [0,0,255]   # red for anomalies
        result = cv2.addWeighted(img, 0.7, overlay, 0.4, 0)

        out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_cnn.png"))
        cv2.imwrite(out_path, result)
        print(fname, "-> saved")

        # ===== Metrics =====
        mask_path = os.path.join(MASK_DIR, fname.replace(".jpg",".png"))
        gt = cv2.imread(mask_path)

        if gt is not None:
            gt_bin = (gt[:,:,2] > 150).astype(np.uint8)
            y_true.extend(gt_bin.flatten())
            y_pred.extend(pred_mask.flatten())

    # ===== Evaluation =====
    if len(y_true) > 0:
        print("\n📊 Evaluation")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    test()
