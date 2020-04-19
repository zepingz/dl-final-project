import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from utils.evaluate import get_mask_threat_score, get_detection_threat_score

class FasterRCNNModel(nn.Module):
    def __init__(self):
        super(FasterRCNNModel, self).__init__()
        self.mask_criterion = nn.BCEWithLogitsLoss()
        
        self.faster_rcnn = fasterrcnn_resnet50_fpn(
            num_classes=9, pretrained_backbone=False)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.norm1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(64, 1, 1, 1, 0)
        self.relu = nn.ReLU()
        
    def get_loss(self, data, device, return_result=False):
        imgs = list(img.to(device) for img in data[0])
        targets = [{k: v.to(device) for k, v in t.items() if k != 'masks'} for t in data[1]]
        target_masks = torch.stack([t['masks'].to(device) for t in data[1]])
        
        # Get loss
        self.train()
        loss_dict = self.faster_rcnn(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        mask = self.img_to_mask(imgs)
        mask_loss = self.mask_criterion(mask, target_masks)
        losses = losses + mask_loss
        
        # Get detection
        self.eval()
        detections = self.faster_rcnn(imgs)
        
        with torch.no_grad():
            # Get mask threat score
            road_imgs = torch.stack([t['masks'] for t in data[2]])
            mask_ts, mask_ts_numerator, mask_ts_denominator = get_mask_threat_score(
                mask.cpu(), road_imgs)
            
            # Get object detection threat score
            cpu_detections = [{k: v.cpu() for k, v in t.items()} for t in detections]
            detection_ts, detection_ts_numerator, detection_ts_denominator =\
                get_detection_threat_score(cpu_detections, data[2], 0.5)
            
        if return_result:
            return losses, detections, masks, mask_ts, mask_ts_numerator,\
                   mask_ts_denominator, detection_ts, detection_ts_numerator,\
                   detection_ts_denominator
        else:
            return losses, mask_ts, mask_ts_numerator, mask_ts_denominator,\
                   detection_ts, detection_ts_numerator, detection_ts_denominator
    
    def img_to_mask(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
            
        x = self.faster_rcnn.backbone(x)['0']
        x = self.relu(self.norm1(self.deconv1(x)))
        x = self.relu(self.norm2(self.deconv2(x)))
        x = self.conv(x)
        return x