import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
IMG_DIR = "dataset/images/"
MASK_DIR = "dataset/masks/"

# Image size and training params
SIZE = 256
BATCH = 4
EPOCHS = 10

# ---------------------------
# Dataset class
# ---------------------------
class LeafDataset(Dataset):
    def __init__(self):
        self.files = os.listdir(IMG_DIR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(IMG_DIR, name))
        mask = cv2.imread(os.path.join(MASK_DIR, name.replace(".jpg",".png")), 0)

        # resize
        img = cv2.resize(img, (SIZE, SIZE))
        mask = cv2.resize(mask, (SIZE, SIZE))

        # normalize
        img = img / 255.0
        mask = mask / 255.0
        mask = (mask > 0).astype(np.float32)  # binary mask

        # to torch tensors
        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()
        return img, mask

# ---------------------------
# Simple U-Net
# ---------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def C(in_c,out_c):
            return nn.Sequential(
                nn.Conv2d(in_c,out_c,3,padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c,out_c,3,padding=1),
                nn.ReLU()
            )
        self.d1 = C(3,64)
        self.d2 = C(64,128)
        self.pool = nn.MaxPool2d(2)
        self.mid = C(128,256)
        self.up = nn.Upsample(scale_factor=2)
        self.u2 = C(256+128,128)
        self.u1 = C(128+64,64)
        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        mid = self.mid(self.pool(d2))
        u2 = self.up(mid)
        u2 = self.u2(torch.cat([u2,d2],1))
        u1 = self.up(u2)
        u1 = self.u1(torch.cat([u1,d1],1))
        return self.out(u1)  # no sigmoid here

# ---------------------------
# Training loop
# ---------------------------
if __name__ == "__main__":
    dataset = LeafDataset()
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        total_loss = 0
        for img, mask in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)
            loss = loss_fn(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "leaf_unet.pth")
    print("Training done! Model saved as leaf_unet.pth")
