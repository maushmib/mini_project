import cv2
import numpy as np
import torch
import torch.nn as nn
import os

PATCH_SIZE = 3
STRIDE = 5

TEST_DIR = "dataset/tests/"
OUTPUT_DIR = "dataset/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# same CNN
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


model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("cnn_model.pth", map_location=DEVICE))
model.eval()


for fname in os.listdir(TEST_DIR):

    img_path = os.path.join(TEST_DIR, fname)
    original_img = cv2.imread(img_path)

    if original_img is None:
        continue

    h, w = original_img.shape[:2]

    scale = 0.5
    img = cv2.resize(original_img, (int(w*scale), int(h*scale)))

    h_small, w_small = img.shape[:2]

    mask_small = np.zeros((h_small, w_small), dtype=np.uint8)

    for y in range(0, h_small-PATCH_SIZE+1, STRIDE):
        for x in range(0, w_small-PATCH_SIZE+1, STRIDE):

            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] / 255.0
            patch = np.transpose(patch, (2,0,1))

            tensor = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                pred = model(tensor).argmax(1).item()

            if pred == 1:
                mask_small[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255


    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    result = original_img.copy()
    result[mask==255] = [0,0,255]

    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg","_output.png"))
    cv2.imwrite(out_path, result)

    print("Saved:", out_path)


print("All test images processed!")
