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
                self.data = json.load(f)
            self.images = self.data["images"]
            self.annotations = self.data["annotations"]
            
            # Create mapping from image id to annotations
            self.img_to_anns = {}
            for ann in self.annotations:
                img_id = ann["image_id"]
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)
                
            print(f"Loaded {len(self.images)} images, {len(self.annotations)} annotations")
        else:
            self.images = [{"file_name": f, "id": i} for i, f in enumerate(os.listdir(img_dir)) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def load_mask(self, img_id, orig_w, orig_h):
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann["segmentation"]:
                    if isinstance(seg, list) and len(seg) >= 6:  # Valid polygon
                        poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                        
                        # Scale coordinates
                        scale_x = self.img_size / orig_w
                        scale_y = self.img_size / orig_h
                        poly[:, 0] *= scale_x
                        poly[:, 1] *= scale_y
                        
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        
        return mask

    def __getitem__(self, idx):
        info = self.images[idx]
        filename = info["file_name"]
        img_path = os.path.join(self.img_dir, filename)

        try:
            # Load and resize image with error handling
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        except Exception as e:
            print(f"❌ Error loading image {filename}: {e}")
            # Return a blank image or skip
            img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            if self.has_labels:
                mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
                return img, mask
            else:
                return img, filename

        if self.has_labels:
            img_id = info["id"]
            mask = self.load_mask(img_id, orig_w, orig_h)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return img, mask
        else:
            return img, filename