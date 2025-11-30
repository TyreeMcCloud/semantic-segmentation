import torch
import torch.nn as nn
import torch.nn.functional as F

# post-processing to remove small speckle blobs
def clean_prediction(pred_bin, min_cluster_size=7, erosion_strength=1):
    # -----------------------------
    # Connected Component Cleanup
    # -----------------------------
    # label connected components with convolution (pseudo-CC)
    kernel = torch.ones((1, 1, 3, 3), device=pred_bin.device)
    neighbor_count = F.conv2d(pred_bin, kernel, padding=1)

    # Keep only pixels belonging to regions >= min_cluster_size
    pred_clean = (neighbor_count >= min_cluster_size).float()

    # -----------------------------
    # Heavy Erosion (shrinks big tumor)
    # -----------------------------
    for _ in range(erosion_strength):
        neighbor_count = F.conv2d(pred_clean, kernel, padding=1)
        # require ≥6 neighbors to survive → strong shrinkage
        pred_clean = (neighbor_count >= 6).float()

    return pred_clean

THRESHOLD = 0.5
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Apply sigmoid if using raw logits
        predictions = torch.sigmoid(predictions)

        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice

def dice_coefficient(predictions, targets, smooth=1e-6):
    pred_binary = (torch.sigmoid(predictions) > THRESHOLD).float()

    #pred_binary = (torch.sigmoid(predictions) > THRESHOLD).float()
    pred_binary = clean_prediction(pred_binary)

    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = targets.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    total_pred = pred_flat.sum()
    total_target = target_flat.sum()

    #print(f"DEBUG: Intersection: {intersection.item()}, Pred sum: {total_pred.item()}, Target sum: {total_target.item()}")

    dice = (2. * intersection + smooth) / (total_pred + total_target + smooth)

    if dice.isnan():
        return 0.0
    return dice.item()

# IoU metric
def iou(pred, mask):
    #pred = (torch.sigmoid(pred) > THRESHOLD).float()

    pred_bin = (torch.sigmoid(pred) > THRESHOLD).float()
    pred = clean_prediction(pred_bin)

    intersection = (pred * mask).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + mask.sum(dim=[1, 2, 3]) - intersection
    iou = (intersection / (union + 1e-6)).mean()
    return iou.item() #+ (1 - (THRESHOLD*2))

# Pixel Accuracy metric
def pixel_accuracy(pred, mask):
    #pred = (torch.sigmoid(pred) > THRESHOLD).float()

    pred_bin = (torch.sigmoid(pred) > THRESHOLD).float()
    pred = clean_prediction(pred_bin)

    correct = (pred == mask).float().sum(dim=[1, 2, 3])
    total = mask[0].numel()
    accuracy = correct.mean() / total
    return accuracy.item()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_sigmoid = torch.sigmoid(pred)
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * focal_loss
        return focal_loss.mean()
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss