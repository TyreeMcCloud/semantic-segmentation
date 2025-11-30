import torch
from torch.utils.data import DataLoader
from dataset import TumorSegmentationDataset
from model import UNet
from metrics import iou, pixel_accuracy, dice_coefficient, clean_prediction, THRESHOLD
from config import VAL_IMG_DIR, VAL_JSON
import torch.nn.functional as F

def evaluate_validation():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load("final_model.pth", map_location=device))
    model.eval()

    # Validation dataset (has ground truth)
    val_ds = TumorSegmentationDataset(VAL_IMG_DIR, VAL_JSON)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    total_iou = 0
    total_acc = 0
    total_dice = 0
    count = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            # Raw model output (logits)
            preds = model(imgs)

            total_iou += iou(preds, masks)
            total_acc += pixel_accuracy(preds, masks)
            total_dice += dice_coefficient(preds, masks)

            count += 1

    print(f"VALIDATION RESULTS:")
    print(f"IoU: {total_iou/count:.4f}")
    print(f"Accuracy: {total_acc/count:.4f}")
    print(f"Dice: {total_dice/count:.4f}")

    if total_iou/count >= 0.70 and total_acc/count >= 0.75:
        print("MINIMUM REQUIREMENTS MET!")
    else:
        print("Minimum requirements NOT met")

if __name__ == "__main__":
    evaluate_validation()
