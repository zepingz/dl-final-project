import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.basic_model import BasicModel
from models.faster_rcnn_model import FasterRCNNModel
from models.new_faster_rcnn import ModifiedFasterRCNN

def get_model(args):
    if args.model == 'basic':
        model = BasicModel()
    elif args.model == 'faster_rcnn':
        model = FasterRCNNModel()
    elif args.model == 'new_faster_rcnn':
        backbone_list = []
        for i in range(6):
            backbone_list.append(resnet_fpn_backbone('resnet50', True))
        model = ModifiedFasterRCNN(backbone_list, num_classes=9)
        
    return model
