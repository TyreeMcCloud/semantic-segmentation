from src.dataset import TumorSegmentationDataset
import numpy as np
ds = TumorSegmentationDataset("Dataset/train", "Dataset/train/_annotations.coco.json", img_size=216)
img, mask = ds[0]
print(img.shape, mask.shape)
print("Mask unique values:", np.unique(mask))
