import torch
import cv2
import numpy as np
from src.dataset import TumorSegmentationDataset
from src.config import TEST_IMG_DIR
from src.model import UNet

model = UNet().cuda()
model.load_state_dict(torch.load("model.pth"))
model.eval()

test_ds = TumorSegmentationDataset(TEST_IMG_DIR)

for img, filename in test_ds:
    img_gpu = img.unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(img_gpu)[0][0].cpu().numpy()

    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    cv2.imwrite(f"pred_masks/{filename}", pred_mask)

print("Saved predictions to pred_masks/")
