import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.basic_model import BasicModel
from models.faster_rcnn_model import FasterRCNNModel

def get_model(args):
    if args.model == 'basic':
        model = BasicModel()
    elif args.model == 'faster_rcnn':
        model = FasterRCNNModel()
        
    return model
