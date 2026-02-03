import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# PATHS
# =====================================================
IMAGE_DIR  = "dataset/images/"
MASK_DIR   = "dataset/masks/"
TEST_DIR   = "dataset/tests/"
OUTPUT_DIR = "dataset/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# SETTINGS
# =====================================================
PATCH_SIZE = 25      # big → learns texture/shape
STRIDE = 4
BATCH = 256
EPOCHS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# DATASET (center labeling)
# =====================================================
class PatchDataset(Dataset):

    def __init__(self):
        self.data = []
        print("Preparing patches...")

        for fname in os.listdir(IMAGE_DIR):

            if not fname.endswith(".jpg"):
                continue

            img  = cv2.imread(os.path.join(IMAGE_DIR, fname))
            mask = cv2.imread(os.path.join(MASK_DIR, fname.replace(".jpg", ".png")))

            h, w = img.shape[:2]

            for y in range(0, h-PATCH_SIZE+1, STRIDE):
                for x in range(0, w-PATCH_SIZE+1, STRIDE):

                    patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                    patch = (patch/255.0).astype(np.float32)
                    patch = np.transpose(patch, (2,0,1))

                    # ⭐ center pixel labeling
                    c = PATCH_SIZE // 2
                    label = 1 if mask_patch[c,c,2] > 150 else 0

                    self.data.append((patch, label))

        print("Total patches:", len(self.data))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)


# =====================================================
# CNN (texture aware)
# =====================================================
class PatchCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        size = PATCH_SIZE // 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*size*size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )


    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# =====================================================
# TRAIN
# =====================================================
def train():

    dataset = PatchDataset()
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model = PatchCNN().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining...")

    for e in range(EPOCHS):

        total_loss = 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {e+1}/{EPOCHS}  Loss: {total_loss:.3f}")

    torch.save(model.state_dict(), "patch_cnn_model.pth")
    print("Model saved -> patch_cnn_model.pth")


# =====================================================
# TEST
# =====================================================
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

        img = cv2.imread(os.path.join(TEST_DIR, fname))
        h, w = img.shape[:2]

        pred_mask = np.zeros((h,w), dtype=np.uint8)

        for y in range(0, h-PATCH_SIZE+1, STRIDE):
            for x in range(0, w-PATCH_SIZE+1, STRIDE):

                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch = (patch/255.0).astype(np.float32)
                patch = np.transpose(patch,(2,0,1))

                tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred = model(tensor).argmax(1).item()

                # ⭐ BIG RED BOX
                if pred == 1:
                    pred_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 1


        # ===== Overlay visualization =====
        overlay = img.copy()
        overlay[pred_mask==1] = [0,0,255]
        result = cv2.addWeighted(img, 0.7, overlay, 0.4, 0)

        out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_cnn.png"))
        cv2.imwrite(out_path, result)

        print(fname, "-> saved")


        # ===== metrics =====
        mask_path = os.path.join(MASK_DIR, fname.replace(".jpg",".png"))
        gt = cv2.imread(mask_path)

        if gt is not None:
            gt_bin = (gt[:,:,2] > 150).astype(np.uint8)
            y_true.extend(gt_bin.flatten())
            y_pred.extend(pred_mask.flatten())


    if len(y_true) > 0:
        print("\n📊 Evaluation")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    train()
    test()
