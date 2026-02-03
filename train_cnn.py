import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

IMAGE_DIR = "dataset/images/"
MASK_DIR = "dataset/masks/"

PATCH_SIZE = 5    # bigger patch
STRIDE = 2        # smaller stride
BATCH = 256
EPOCHS = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Dataset: extract small patches
# -------------------------
class PatchDataset(Dataset):
    def __init__(self):
        self.data = []

        for fname in os.listdir(IMAGE_DIR):
            if not fname.endswith(".jpg"):
                continue

            img = cv2.imread(os.path.join(IMAGE_DIR, fname))
            mask = cv2.imread(os.path.join(MASK_DIR, fname.replace(".jpg", ".png")))

            if img is None or mask is None:
                continue

            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))

            h, w = img.shape[:2]

            for y in range(0, h-PATCH_SIZE+1, STRIDE):
                for x in range(0, w-PATCH_SIZE+1, STRIDE):
                    patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] / 255.0
                    patch = np.transpose(patch, (2,0,1))

                    mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    label = 1 if np.any(mask_patch[:,:,2] > 150) else 0

                    self.data.append((patch, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x).float(), torch.tensor(y).long()


# -------------------------
# Slightly bigger CNN
# -------------------------
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


# -------------------------
# Training
# -------------------------
print("Preparing patches...")
dataset = PatchDataset()
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

model = PatchCNN().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training CNN...")
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

    print(f"Epoch {e+1} Loss: {total_loss:.3f}")

torch.save(model.state_dict(), "cnn_model_fixed.pth")
print("Model saved as cnn_model_fixed.pth")
