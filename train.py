import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.config import *
from src.dataset import TumorSegmentationDataset
from src.model import UNet
from src.metrics import iou, pixel_accuracy, DiceLoss, dice_coefficient, FocalLoss

def main():
    # Set device for M1 Mac
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = TumorSegmentationDataset(TRAIN_IMG_DIR, TRAIN_JSON)

    # Quick check to ensure masks are being loaded correctly
    print("Checking if masks are loaded correctly...")
    for i in range(3):
        img, mask = train_ds[i]
        tumor_pixels = mask.sum().item()
        print(f"Sample {i}: Tumor pixels: {tumor_pixels}")

        if tumor_pixels > 0:
            print("Tumors found in masks")
            break
    else:
        print("Still no tumrs - figure out why annotations aren't loading correctly.\n")

    val_ds = TumorSegmentationDataset(VAL_IMG_DIR, VAL_JSON)
    print(f"Datasets loaded: {len(train_ds)} training samples, {len(val_ds)} validation samples.")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = UNet().to(device)

    #loss_fn = DiceLoss()
    #loss_fn = nn.BCEWithLogitsLoss()
    # pos_weight = torch.tensor([20.0]).to(device)
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn = FocalLoss(alpha=0.9, gamma=1.5)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_iou = 0.0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # Training
        model.train()
        total_loss = 0
        train_batches = 0

        for imgs, masks in train_loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            # Add inside your training loop, after preds = model(imgs)
            # if epoch == 0 and train_batches == 0:  # Only first batch of first epoch
            #   print(f"First batch - Pred range: [{preds.min().item():.3f}, {preds.max().item():.3f}]")
            #   print(f"First batch - After sigmoid: [{torch.sigmoid(preds).min().item():.3f}, {torch.sigmoid(preds).max().item():.3f}]")

            if train_batches == 0:  # First batch of every epoch
                print(f"Epoch {epoch+1} - Pred range: [{preds.min().item():.3f}, {preds.max().item():.3f}]")
                print(f"Epoch {epoch+1} - After sigmoid: [{torch.sigmoid(preds).min().item():.3f}, {torch.sigmoid(preds).max().item():.3f}]")

            # see what's happening:
            if epoch == 0 and train_batches == 0:
                print(f"Actual tumor % in batch: {(masks.sum() / masks.numel() * 100):.2f}%")
                print(f"Predicted tumor %: {(torch.sigmoid(preds) > 0.3).float().mean().item() * 100:.2f}%")

            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
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

        # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    print(f"saved as final_model.pth")

    print(f"Done Training. Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()