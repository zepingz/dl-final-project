from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict, Optional

import torchvision.transforms as transforms
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList

from utils.evaluate import get_mask_threat_score, get_detection_threat_score

MEAN = [0.5459, 0.5968, 0.6303] # [0.485, 0.456, 0.406]
STD = [0.3178, 0.3246, 0.3278] # [0.229, 0.224, 0.225]

class DetectionGeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform, input_img_num=6):
        super(DetectionGeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.input_img_num = input_img_num
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.backbone_out_channels = backbone.out_channels
        
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.target_transform = GeneralizedRCNNTransform(
            400, 400, [0., 0., 0.], [1., 1., 1.])

    def forward(self, _images, targets=None, return_result=False):
        bs = _images.size(0)
        assert bs == 1
        
        
        # Process images
        device = _images.device
        images = torch.zeros(1, 6, 3, 400, 400)
        for i in range(6):
            images[0, i] = self.img_transform(_images[0, i].cpu())
        del _images
        images = images.to(device)
        
        # Process targets
#         label_index = targets[0]['labels'] == 2
#         targets[0]['boxes'] = targets[0]['boxes'][label_index]
#         targets[0]['labels'] = targets[0]['labels'][label_index]
        
        targets = [{k: v for k, v in t.items()} for t in targets]
        targets[0]['old_boxes'] = targets[0]['boxes'] / 2.
        min_coordinates, _ = torch.min(targets[0]['boxes'], 2)
        max_coordinates, _ = torch.max(targets[0]['boxes'], 2)
        targets[0]['boxes'] = torch.cat([min_coordinates, max_coordinates], 1)
        temp_tensor = torch.zeros(1, 3, 800, 800)
        _, targets = self.target_transform(temp_tensor, targets)
            
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # images, targets = self.transform(images, targets)
        # HACK
        images = ImageList(images, ((400, 400),) * images.size(0))
        targets = [{k: v.to(images.tensors.device) for k, v in t.items() if k != 'masks'} for t in targets]

        # Pass images from 6 camera angle to different backbone
        features_list = torch.stack(
            [self.backbone(images.tensors[:, i])['0'] for i in range(self.input_img_num)], dim=1)

        feature_h, feature_w = features_list.size()[-2:]
        features_list = features_list.view(
            bs, self.backbone_out_channels * self.input_img_num, feature_h, feature_w)

        features = OrderedDict([('0', features_list)])
#         if isinstance(features, torch.Tensor):
#             features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update({'loss_mask': torch.zeros(1, device=images.tensors.device)})

        mask_ts = 0.
        mask_ts_numerator = 0
        mask_ts_denominator = 1

        with torch.no_grad():

            # Get object detection threat score
            cpu_detections = [{k: v.cpu() for k, v in t.items()} for t in detections]
            # TODO: add threshold more than 0.5
            detection_ts, detection_ts_numerator, detection_ts_denominator =\
                get_detection_threat_score(cpu_detections, targets, 0.5)

        if return_result:
            # DEBUG
            masks = 0
#             return losses, mask_ts, mask_ts_numerator,\
#                    mask_ts_denominator, detection_ts, detection_ts_numerator,\
#                    detection_ts_denominator, detections, masks
            return mask_ts, mask_ts_numerator,\
                   mask_ts_denominator, detection_ts, detection_ts_numerator,\
                   detection_ts_denominator, detections, masks
        else:
#             return losses, mask_ts, mask_ts_numerator, mask_ts_denominator,\
#                    detection_ts, detection_ts_numerator, detection_ts_denominator
            return losses

class DetectionFasterRCNN(DetectionGeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.5,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.5, box_nms_thresh=0.3, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

#         if not hasattr(backbone, "out_channels"):
#             raise ValueError(
#                 "backbone should contain an attribute out_channels "
#                 "specifying the number of output channels (assumed to be the "
#                 "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = 256 * 6 # backbone.out_channels

        if rpn_anchor_generator is None:
            # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            anchor_sizes = ((8,), (16,),)
            aspect_ratios = ((0.5, 0.7, 1.0,),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                # featmap_names=['0', '1', '2', '3'],
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        # mask_net = MaskNet(out_channels)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(DetectionFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
