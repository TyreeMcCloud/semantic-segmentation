# test_fix.py
from src.dataset import TumorSegmentationDataset
import json

# Check the COCO structure
with open("Dataset/train/_annotations.coco.json", "r") as f:
    data = json.load(f)

print("COCO Structure Check:")
print(f"First image: {data['images'][0]}")
print(f"First annotation: {data['annotations'][0]}")

# Test the fixed dataset
train_ds = TumorSegmentationDataset("Dataset/train/", "Dataset/train/_annotations.coco.json")

tumor_counts = []
for i in range(10):
    img, mask = train_ds[i]
    tumor_pixels = mask.sum().item()
    tumor_counts.append(tumor_pixels)
    print(f"Sample {i}: {tumor_pixels} tumor pixels")

print(f"Samples with tumors: {sum(1 for x in tumor_counts if x > 0)}/10")