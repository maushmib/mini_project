import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import PatchCNN, PATCH_SIZE, STRIDE

# ===== Paths =====
TEST_DIR   = "dataset/tests/"
MASK_DIR   = "dataset/masks/"
OUTPUT_DIR = "dataset/output/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# threshold for hybrid detection
HYBRID_RATIO_THRESHOLD = 0.01   # 1%


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

                c = PATCH_SIZE // 2

                if pred == 1:
                    cv2.circle(pred_mask, (x+c, y+c), 2, 1, -1)

        # ===== Remove noise =====
        kernel = np.ones((5,5), np.uint8)

        clean_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)

        pred_mask = clean_mask

        # ===== Calculate hybrid ratio using pixels =====
        hybrid_pixels = np.sum(pred_mask == 1)
        total_pixels = h * w

        ratio = hybrid_pixels / total_pixels

        print(f"{fname} -> hybrid pixels: {hybrid_pixels}")
        print(f"Hybrid ratio: {ratio:.4f}")

        if ratio > HYBRID_RATIO_THRESHOLD:
            print("Hybrid plant detected\n")
        else:
            print("Normal plant\n")

        # ===== Save overlay image =====
        overlay = img.copy()
        overlay[pred_mask == 1] = [0,0,255]

        result = cv2.addWeighted(img, 0.7, overlay, 0.4, 0)

        out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_cnn.png"))
        cv2.imwrite(out_path, result)

        print(fname, "-> saved")

        # ===== Accuracy check if mask exists =====
        mask_name = fname.replace(".jpg", ".png")
        mask_path = os.path.join(MASK_DIR, mask_name)

        if os.path.exists(mask_path):

            gt = cv2.imread(mask_path)

            gt_bin = (gt[:,:,2] > 150).astype(np.uint8)

            y_true.extend(gt_bin.flatten())
            y_pred.extend(pred_mask.flatten())

        else:
            print(f"⚠ No mask for {fname} → skipping accuracy")

    # ===== Evaluation =====
    if len(y_true) > 0:

        print("\n📊 Evaluation")

        acc = accuracy_score(y_true, y_pred)
        print("Accuracy:", acc)

        report = classification_report(y_true, y_pred)
        print(report)

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)

        # ===== Plot Confusion Matrix =====
        plt.figure(figsize=(5,4))

        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = ["Normal", "Hybrid"]
        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="black")

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        plt.tight_layout()

        plt.savefig("confusion_matrix.png")
        plt.show()

    else:
        print("\n(No masks found → Accuracy skipped)")


if __name__ == "__main__":
    test()