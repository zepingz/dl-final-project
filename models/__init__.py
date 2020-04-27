import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .basic_model import BasicModel
from .faster_rcnn_model import FasterRCNNModel
from .new_faster_rcnn import ModifiedFasterRCNN
from .detection_faster_rcnn import DetectionFasterRCNN

# from torchvision.ops import misc as misc_nn_ops
# from torchvision.models import resnet
# from torchvision.models.detection.backbone_utils import BackboneWithFPN
# def resnet_fpn_backbone(backbone_name, pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d):
#     backbone = resnet.__dict__[backbone_name](
#         pretrained=pretrained,
#         norm_layer=norm_layer)
#     # freeze layers
#     for name, parameter in backbone.named_parameters():
#         if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#             parameter.requires_grad_(False)
#
#     return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
#
#     in_channels_stage2 = backbone.inplanes // 8
#     in_channels_list = [
#         in_channels_stage2,
#         in_channels_stage2 * 2,
#         in_channels_stage2 * 4,
#         in_channels_stage2 * 8,
#     ]
#     out_channels = 128
#     return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

def get_model(args):
    if args.model == 'basic':
        model = BasicModel()
    elif args.model == 'faster_rcnn':
        model = FasterRCNNModel()
    elif args.model == 'new_faster_rcnn':
        # backbone_list = []
        # for i in range(6):
        #     backbone_list.append(resnet_fpn_backbone('resnet50', False))
        # model = ModifiedFasterRCNN(backbone_list, num_classes=9)
        model = ModifiedFasterRCNN(resnet_fpn_backbone('resnet18', False), num_classes=9)
    elif args.model == 'detection_faster_rcnn':
        model = DetectionFasterRCNN(resnet_fpn_backbone('resnet18', False), num_classes=9)

    return model
