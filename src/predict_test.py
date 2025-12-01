import torch
import cv2
import numpy as np
import os
from dataset import TumorSegmentationDataset
from config import TEST_IMG_DIR
from model import UNet
from metrics import clean_prediction, THRESHOLD

def predict_test_set():

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs("pred_masks", exist_ok=True)

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load("final_model.pth", map_location=device))
    model.eval()


    test_ds = TumorSegmentationDataset(TEST_IMG_DIR)

    for i in range(len(test_ds)):
        img, filename = test_ds[i]
        img_device = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_device)                # raw logits
            sig = torch.sigmoid(pred)               # sigmoid
            pred_bin = (sig > THRESHOLD).float()    # thresholded
            pred_clean = clean_prediction(pred_bin)

        pred_mask = (pred_clean[0][0].cpu().numpy() * 255).astype(np.uint8)

        # Save predicted mask
        output_path = f"pred_masks/{filename}"
        cv2.imwrite(output_path, pred_mask)
        print(f"Saved: {output_path}")

    print(f"Saved {len(test_ds)} predictions to pred_masks/")

if __name__ == "__main__":
    predict_test_set()