import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import json

class TumorSegmentationDataset(Dataset):
    def __init__(self, img_dir, json_path=None, img_size=256):
        self.img_dir = img_dir
        self.img_size = img_size

        self.has_labels = json_path is not None

        if self.has_labels:
            with open(json_path, "r") as f:
                data = json.load(f)

            self.images = data["images"]
            self.annotations = data["annotations"]
        else:
            self.images = [
                {"file_name": f} for f in os.listdir(img_dir) if f.endswith(".jpg")
            ]

    def __len__(self):
        return len(self.images)

    def load_mask(self, img_id, orig_w, orig_h):
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        anns = [a for a in self.annotations if a["image_id"] == img_id]

        for ann in anns:
            for seg in ann["segmentation"]:
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2)

                # SCALE POLYGON COORDS
                scale_x = self.img_size / orig_w
                scale_y = self.img_size / orig_h
                poly[:, 0] *= scale_x
                poly[:, 1] *= scale_y

                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        # Resize mask to match model size
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        #print("Mask unique:", np.unique(mask))
        return mask

    def __getitem__(self, idx):
        info = self.images[idx]
        file = info["file_name"]
        img_path = os.path.join(self.img_dir, file)

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize image
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)

        if self.has_labels:
            img_id = info["id"]
            mask = self.load_mask(img_id, orig_w, orig_h)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.has_labels:
            return img, mask
        else:
            return img, file
