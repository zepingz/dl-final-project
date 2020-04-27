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

def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull
    
    return a.intersection(b).area / a.union(b).area

def get_mask_threat_score(pred, target):
#     resize_transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((800, 800)),
#         transforms.ToTensor()
#     ])
    
    pred_road_imgs = torch.stack([
        (nn.Sigmoid()(pred_img) > 0.5).int() for pred_img in pred])
    
#     if pred_road_imgs.shape[2] != 800 or pred_road_imgs.shape[3] != 800:
#         pred_road_imgs = torch.stack([resize_transform(img)\
#             for img in pred_road_imgs])
        
    tp = torch.sum((pred_road_imgs > 0.) * (target > 0.)).item()
    fp = torch.sum((pred_road_imgs > 0.) * (target < 1.)).item()
    fn = torch.sum((pred_road_imgs < 1.) * (target > 0.)).item()
    ts_denominator = tp + fp + fn
    try:
        ts = tp / ts_denominator
    except ZeroDivisionError:
        ts = 0.
    return ts, tp, ts_denominator

def compute_ats_bounding_boxes(boxes1, boxes2):
    num_boxes1 = boxes1.size(0)
    num_boxes2 = boxes2.size(0)

    boxes1_max_x = boxes1[:, 0].max(dim=1)[0]
    boxes1_min_x = boxes1[:, 0].min(dim=1)[0]
    boxes1_max_y = boxes1[:, 1].max(dim=1)[0]
    boxes1_min_y = boxes1[:, 1].min(dim=1)[0]

    boxes2_max_x = boxes2[:, 0].max(dim=1)[0]
    boxes2_min_x = boxes2[:, 0].min(dim=1)[0]
    boxes2_max_y = boxes2[:, 1].max(dim=1)[0]
    boxes2_min_y = boxes2[:, 1].min(dim=1)[0]

    condition1_matrix = (boxes1_max_x.unsqueeze(1) > boxes2_min_x.unsqueeze(0))
    condition2_matrix = (boxes1_min_x.unsqueeze(1) < boxes2_max_x.unsqueeze(0))
    condition3_matrix = (boxes1_max_y.unsqueeze(1) > boxes2_min_y.unsqueeze(0))
    condition4_matrix = (boxes1_min_y.unsqueeze(1) < boxes2_max_y.unsqueeze(0))
    condition_matrix = condition1_matrix * condition2_matrix * condition3_matrix * condition4_matrix

    iou_matrix = torch.zeros(num_boxes1, num_boxes2)
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            if condition_matrix[i][j]:
                # iou_matrix[i][j] = compute_iou(boxes1[i], boxes2[j])
                iou_matrix[i][j] = box_iou(boxes1[i].view(1, 4), boxes2[j].view(1, 4))

    iou_max = iou_matrix.max(dim=0)[0]

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    total_threat_score = 0
    total_weight = 0
    for threshold in iou_thresholds:
        tp = (iou_max > threshold).sum()
        threat_score = tp * 1.0 / (num_boxes1 + num_boxes2 - tp)
        total_threat_score += 1.0 / threshold * threat_score
        total_weight += 1.0 / threshold

    average_threat_score = total_threat_score / total_weight
    
    return average_threat_score, total_threat_score, total_weight

def get_detection_threat_score(pred, target, threshold):
    # target_transform = TargetResize((512, 920), (800, 800))
    
    tp, fp, fn = 0, 0, 0
    for pred_datapoint, gt_datapoint in zip(pred, target):
        # pred_datapoint = target_transform(pred_datapoint)
        if isinstance(pred_datapoint['boxes'], list) and len(pred_datapoint['boxes']) == 0:
            pred_match, gt_match = 0, 0
        else:
            pred_boxes = pred_datapoint['boxes']
            gt_boxes = gt_datapoint['boxes'].cpu()
            iou = box_iou(pred_boxes, gt_boxes)
            pred_match, gt_match = torch.where(iou > threshold)
            gt_match = len(torch.unique(gt_match))
            pred_match = len(torch.unique(pred_match))
        tp += gt_match
        fp += len(pred_boxes) - pred_match
        fn += len(gt_boxes) - gt_match
        
    ts_denominator = tp + fp + fn
    try:
        ts = tp / ts_denominator
    except ZeroDivisionError:
        ts = 0.

    return ts, tp, ts_denominator
