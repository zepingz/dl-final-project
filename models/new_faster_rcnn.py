from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict, Optional

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList

from utils.evaluate import get_mask_threat_score, get_detection_threat_score

class MaskNet(nn.Module):
    def __init__(self, in_channels):
        super(MaskNet, self).__init__()
        self.in_channels = in_channels
        
        self.deconv1 = nn.ConvTranspose2d(
            self.in_channels, 512, 2, stride=2, padding=0)
        self.deconv1_norm = nn.BatchNorm2d(512)
        # self.conv1_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(self.in_channels, 512, 3, stride=1, padding=1)
        self.conv1_1_norm = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv1_2_norm = nn.BatchNorm2d(512)
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        self.deconv2_norm = nn.BatchNorm2d(256)
        # self.conv2_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv2_1_norm = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2_2_norm = nn.BatchNorm2d(256)
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.deconv3_norm = nn.BatchNorm2d(128)
        # self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv3_1_norm = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_2_norm = nn.BatchNorm2d(128)
        self.conv_last = nn.Conv2d(128, 1, 1, 1, 0)
        self.relu = nn.ReLU()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, target_masks=None):
        device = x.device
        
        # x = self.relu(self.deconv1_norm(self.deconv1(x)))
        x = self.up(x)
        x = self.relu(self.conv1_1_norm(self.conv1_1(x)))
        x = self.relu(self.conv1_2_norm(self.conv1_2(x)))
        
        # x = self.relu(self.deconv2_norm(self.deconv2(x)))
        x = self.up(x)
        x = self.relu(self.conv2_1_norm(self.conv2_1(x)))
        x = self.relu(self.conv2_2_norm(self.conv2_2(x)))
        
        # x = self.relu(self.deconv3_norm(self.deconv3(x)))
        x = self.up(x)
        x = self.relu(self.conv3_1_norm(self.conv3_1(x)))
        x = self.relu(self.conv3_2_norm(self.conv3_2(x)))
        x = self.conv_last(x)
        
        if target_masks is not None:
            losses = {
                'loss_mask': nn.BCEWithLogitsLoss()(x, target_masks)
            }
            return x, losses
        else:
            return x

class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone_list, rpn, roi_heads, mask_net, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone_list = nn.ModuleList([b for b in backbone_list])
        self.backbone_num = len(backbone_list)
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.mask_net = mask_net
        self.backbone_out_channels = backbone_list[0].out_channels

    def forward(self, images, targets=None, return_result=False):
        assert images.size(1) == self.backbone_num
        bs = images.size(0)
        
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # images, targets = self.transform(images, targets)
        # HACK
        device = images.device
        images = ImageList(images, ((800, 800),) * images.size(0))
        target_masks = torch.stack([t['masks'].to(device) for t in targets])
        targets = [{k: v.to(device) for k, v in t.items() if k != 'masks'} for t in targets]
        
        # Pass images from 6 camera angle to different backbone
        features_list = []
        for i in range(self.backbone_num):
            features_list.append(self.backbone_list[i](images.tensors[:, i])['0'])
            
        features_list = torch.stack(features_list, dim=1)
        feature_h, feature_w = features_list.size()[-2:]
        features_list = features_list.view(
            bs, self.backbone_out_channels * self.backbone_num, feature_h, feature_w)
        
        masks, mask_losses = self.mask_net(features_list, target_masks)
        
        # DEBUG
        losses = {
            'loss_rpn_box_reg': torch.tensor([0.]).cuda(),
            'loss_objectness': torch.tensor([0.]).cuda(),
            'loss_box_reg': torch.tensor([0.]).cuda(),
            'loss_classifier': torch.tensor([0.]).cuda(),
        }
        
        # TODO: Add something like unet
        # TODO: Add a branch to predict the angle
        # TODO: See if using a shared upsampling network is good or not
        
#         features = OrderedDict([('0', features_list)])
# #         if isinstance(features, torch.Tensor):
# #             features = OrderedDict([('0', features)])
        
#         proposals, proposal_losses = self.rpn(images, features, targets)
#         detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
#         detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)
        losses.update(mask_losses)
        
#         mask_ts = 0.
#         mask_ts_numerator = 0
#         mask_ts_denominator = 1
        detection_ts = 0.
        detection_ts_numerator = 0
        detection_ts_denominator = 1
        
        with torch.no_grad():
            # Get mask threat score
            mask_ts, mask_ts_numerator, mask_ts_denominator = get_mask_threat_score(
                masks.cpu(), target_masks.cpu())
            
#             # Get object detection threat score
#             cpu_detections = [{k: v.cpu() for k, v in t.items()} for t in detections]
#             # TODO: add threshold more than 0.5
#             detection_ts, detection_ts_numerator, detection_ts_denominator =\
#                 get_detection_threat_score(cpu_detections, targets, 0.5)
            
        if return_result:
            # DEBUG
            detections = 0
            return losses, mask_ts, mask_ts_numerator,\
                   mask_ts_denominator, detection_ts, detection_ts_numerator,\
                   detection_ts_denominator, detections, masks,
        else:
            return losses, mask_ts, mask_ts_numerator, mask_ts_denominator,\
                   detection_ts, detection_ts_numerator, detection_ts_denominator

class ModifiedFasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone_list, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
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
        # out_channels = 256

        if rpn_anchor_generator is None:
            # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            anchor_sizes = ((8,), (16,), (32,),)
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
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
        
        mask_net = MaskNet(out_channels)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(ModifiedFasterRCNN, self).__init__(backbone_list, rpn, roi_heads, mask_net, transform)


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