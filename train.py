import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.config import *
from src.dataset import TumorSegmentationDataset
from src.model import UNet
from src.metrics import iou, pixel_accuracy

def main():

    train_ds = TumorSegmentationDataset(TRAIN_IMG_DIR, TRAIN_JSON)
    val_ds   = TumorSegmentationDataset(VAL_IMG_DIR, VAL_JSON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1)

    model = UNet().cuda()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs = imgs.cuda()
            masks = masks.cuda()

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_iou = 0
        val_acc = 0
        count = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.cuda()
                masks = masks.cuda()

                preds = model(imgs)

                val_iou += iou(preds, masks)
                val_acc += pixel_accuracy(preds, masks)
                count += 1

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | IoU: {val_iou/count:.4f} | Acc: {val_acc/count:.4f}")

        torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
