import torch
import torch.nn as nn

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
    """Dice coefficient metric (higher is better)"""
    predictions = (torch.sigmoid(predictions) > 0.5).float()
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    return dice.item()

# IoU metric
def iou(pred, mask):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * mask).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + mask.sum(dim=[1, 2, 3]) - intersection
    iou = (intersection / (union + 1e-6)).mean()
    return iou.item()
# Pixel Accuracy metric
def pixel_accuracy(pred, mask):
    pred = (torch.sigmoid(pred) > 0.5).float()
    correct = (pred == mask).float().sum(dim=[1, 2, 3])
    total = mask[0].numel()
    accuracy = correct.mean() / total
    return accuracy.item()
