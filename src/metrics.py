import torch

def iou(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection
    return (intersection / (union + 1e-6)).item()

def pixel_accuracy(pred, mask):
    pred = (pred > 0.5).float()
    correct = (pred == mask).float().sum()
    total = mask.numel()
    return (correct / total).item()
