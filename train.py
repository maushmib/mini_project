import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
import joblib

IMAGE_DIR = "dataset/images/"
MASK_DIR = "dataset/masks/"

PATCH_SIZE = 3
STRIDE = 5

def extract_features_patch(img_patch):
    color_mean = img_patch.mean(axis=(0,1)).tolist()
    gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    return color_mean + [contrast, homogeneity, energy, correlation]

X = []
y_labels = []

print("Loading training data...")

for fname in os.listdir(IMAGE_DIR):
    if not fname.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, fname)
    mask_path = os.path.join(MASK_DIR, fname.replace(".jpg", ".png"))

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if img is None or mask is None:
        continue

    # resize to half
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))

    h, w = img.shape[:2]

    for y in range(0, h-PATCH_SIZE+1, STRIDE):
        for x in range(0, w-PATCH_SIZE+1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            feat = extract_features_patch(patch)
            X.append(feat)

            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            label = 1 if np.any(mask_patch[:,:,2] > 150) else 0
            y_labels.append(label)

X = np.array(X)
y_labels = np.array(y_labels)

print("Training samples:", len(X))

model = RandomForestClassifier(n_estimators=20, max_depth=12, n_jobs=-1)
print("Training Random Forest...")
model.fit(X, y_labels)

joblib.dump(model, "offtype_model.pkl")
print("Training completed and saved as offtype_model.pkl")
