import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load trained Random Forest model
model = joblib.load("offtype_model.pkl")

PATCH_SIZE = 3
STRIDE = 5
TEST_DIR = "dataset/tests/"
GT_DIR = "dataset/ground_truth/"  # Folder with ground truth masks
OUTPUT_DIR = "dataset/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features_patch(img_patch):
    # Color features
    color_mean = img_patch.mean(axis=(0,1)).tolist()
    
    # Texture features (GLCM)
    gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]

    return color_mean + [contrast, homogeneity, energy, correlation]

# Metrics accumulators
all_gt_pixels = []
all_pred_pixels = []

for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(TEST_DIR, fname)
    original_img = cv2.imread(img_path)
    h, w = original_img.shape[:2]

    # Load corresponding ground truth mask
    gt_path = os.path.join(GT_DIR, fname.replace(".jpg", ".png"))
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"No ground truth for {fname}, skipping metrics.")
        continue

    # Resize image for faster processing
    scale = 0.5
    img = cv2.resize(original_img, (int(w*scale), int(h*scale)))
    h_small, w_small = img.shape[:2]
    mask_small = np.zeros((h_small, w_small), dtype=np.uint8)

    # Predict patches
    for y in range(0, h_small-PATCH_SIZE+1, STRIDE):
        for x in range(0, w_small-PATCH_SIZE+1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            feat = extract_features_patch(patch)
            pred = model.predict([feat])[0]
            if pred == 1:
                mask_small[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255

    # Resize mask to original size
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Save output image with red markings
    result = original_img.copy()
    result[mask==255] = [0,0,255]
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_output.png"))
    cv2.imwrite(out_path, result)
    print("Saved:", out_path)

    # Flatten for metrics computation
    all_gt_pixels.extend((gt_mask > 127).astype(int).flatten())  # binary: 0 or 1
    all_pred_pixels.extend((mask > 127).astype(int).flatten())

# Compute metrics
accuracy = accuracy_score(all_gt_pixels, all_pred_pixels)
precision = precision_score(all_gt_pixels, all_pred_pixels)
recall = recall_score(all_gt_pixels, all_pred_pixels)
f1 = f1_score(all_gt_pixels, all_pred_pixels)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nAll test images processed: entire detected leaves are marked in red!")
