import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# PATHS
IMAGE_DIR  = "dataset/images/"
MASK_DIR   = "dataset/masks/"

PATCH_SIZE = 25
STRIDE = 4


# =====================================================
# DATASET
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
# CNN
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
