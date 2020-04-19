import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.target_transforms import TargetResize
        
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def get_mask_threat_score(pred, target):
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800)),
        transforms.ToTensor()
    ])
    
    pred_road_imgs = torch.stack([
        (nn.Sigmoid()(pred_img) > 0.5).int() for pred_img in pred])
    
    if pred_road_imgs.shape[2] != 800 or pred_road_imgs.shape[3] != 800:
        pred_road_imgs = torch.stack([resize_transform(img)\
            for img in pred_road_imgs])
        
    tp = torch.sum((pred_road_imgs > 0.) * (target > 0.)).item()
    fp = torch.sum((pred_road_imgs > 0.) * (target < 1.)).item()
    fn = torch.sum((pred_road_imgs < 1.) * (target > 0.)).item()
    ts_denominator = tp + fp + fn
    try:
        ts = tp / ts_denominator
    except ZeroDivisionError:
        ts = 0.
    return ts, tp, ts_denominator

def get_detection_threat_score(pred, target, threshold):
    target_transform = TargetResize((512, 920), (800, 800))
    
    tp, fp, fn = 0, 0, 0
    for pred_datapoint, gt_datapoint in zip(pred, target):
        pred_datapoint = target_transform(pred_datapoint)
        if isinstance(pred_datapoint['boxes'], list) and len(pred_datapoint['boxes']) == 0:
            pred_match, gt_match = 0, 0
        else:
            iou = box_iou(pred_datapoint['boxes'], gt_datapoint['boxes'])
            pred_match, gt_match = torch.where(iou > threshold)
            gt_match = len(torch.unique(gt_match))
            pred_match = len(torch.unique(pred_match))
        tp += gt_match
        fp += len(pred_datapoint['boxes']) - pred_match
        fn += len(gt_datapoint['boxes']) - gt_match
        
    ts_denominator = tp + fp + fn
    try:
        ts = tp / ts_denominator
    except ZeroDivisionError:
        ts = 0.

    return ts, tp, ts_denominator
