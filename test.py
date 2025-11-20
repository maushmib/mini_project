import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib
import os

# Load trained Random Forest model
model = joblib.load("offtype_model.pkl")

PATCH_SIZE = 3
STRIDE = 5
TEST_DIR = "dataset/tests/"
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

for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(TEST_DIR, fname)
    original_img = cv2.imread(img_path)  # keep original colors
    h, w = original_img.shape[:2]

    # Smaller copy for faster processing
    scale = 0.5
    img = cv2.resize(original_img, (int(w*scale), int(h*scale)))
    h_small, w_small = img.shape[:2]

    mask_small = np.zeros((h_small, w_small), dtype=np.uint8)

    # Process patches for feature-based prediction
    for y in range(0, h_small-PATCH_SIZE+1, STRIDE):
        for x in range(0, w_small-PATCH_SIZE+1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            feat = extract_features_patch(patch)
            pred = model.predict([feat])[0]
            if pred == 1:
                # Mark the entire patch instead of scribbles
                mask_small[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255

    # Resize mask back to original image size
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Apply red marking directly on original image
    result = original_img.copy()
    result[mask==255] = [0,0,255]  # BGR red

    # Save output
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_output.png"))
    cv2.imwrite(out_path, result)
    print("Saved:", out_path)

print("All test images processed: entire detected leaves are marked in red!")
