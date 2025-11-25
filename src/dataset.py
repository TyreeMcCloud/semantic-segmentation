import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import json

class TumorSegmentationDataset(Dataset):
    def __init__(self, img_dir, json_path=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.has_labels = json_path is not None
        self.annotations = None

        if self.has_labels:
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)

            self.image_info = self.annotations["images"]
            self.ann = self.annotations["annotations"]
        else:
            self.image_files = [x for x in os.listdir(img_dir) if x.endswith(".jpg")]

    def __len__(self):
        if self.has_labels:
            return len(self.image_info)
        else:
            return len(self.image_files)

    def load_mask(self, img_id, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in self.ann:
            if ann["image_id"] == img_id:
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        return mask

    def __getitem__(self, idx):
        if self.has_labels:
            info = self.image_info[idx]
            img_path = os.path.join(self.img_dir, info["file_name"])
        else:
            filename = self.image_files[idx]
            img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert("RGB")
        image = image.resize((640, 640))
        image = np.array(image)

        if self.has_labels:
            img_id = info["id"]
            mask = self.load_mask(img_id, 640, 640)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.has_labels:
            return image, mask
        else:
            return image, filename