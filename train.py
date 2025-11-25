import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.config import *
from src.dataset import TumorSegmentationDataset
from src.model import UNet
from src.metrics import iou, pixel_accuracy, DiceLoss, dice_coefficient

def main():
    # Set device for M1 Mac
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = TumorSegmentationDataset(TRAIN_IMG_DIR, TRAIN_JSON)
    val_ds = TumorSegmentationDataset(VAL_IMG_DIR, VAL_JSON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Move model to M1 GPU (MPS)
    model = UNet().to(device)  # CHANGED: .cuda() -> .to(device)

    # Use Dice Loss instead of BCE
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    best_iou = 0.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        train_batches = 0

        for imgs, masks in train_loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_loss / train_batches

        # Validation
        model.eval()
        val_iou = 0
        val_acc = 0
        val_dice = 0
        val_batches = 0

        with torch.no_grad():
            for imgs, masks in val_loader:

                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)

                val_iou += iou(preds, masks)
                val_acc += pixel_accuracy(preds, masks)
                val_dice += dice_coefficient(preds, masks)
                val_batches += 1

        avg_val_iou = val_iou / val_batches
        avg_val_acc = val_acc / val_batches
        avg_val_dice = val_dice / val_batches

        # Update learning rate based on IoU
        scheduler.step(avg_val_iou)

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "best_model.pth")
            print(f" New best model saved with IoU: {best_iou:.4f}")

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"IoU: {avg_val_iou:.4f} | "
              f"Acc: {avg_val_acc:.4f} | "
              f"Dice: {avg_val_dice:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"🎯 Training completed. Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()