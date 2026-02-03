import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import PatchDataset, PatchCNN

BATCH = 256
EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    print("\n✅ Model saved -> patch_cnn_model.pth")


if __name__ == "__main__":
    train()
