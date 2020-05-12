import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict, Optional

import torchvision.transforms as transforms
from torchvision.ops import misc as misc_nn_ops
# from torchvision.ops import MultiScaleRoIAlign
from .poolers import MultiScaleRoIAlign

# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList

from utils.evaluate import get_mask_threat_score, get_detection_threat_score, compute_ats_bounding_boxes
from .unet import UNet
from LKVOLearner.networks import VggDepthEstimator

MEAN = [0.5459, 0.5968, 0.6303] # [0.485, 0.456, 0.406]
STD = [0.3178, 0.3246, 0.3278] # [0.229, 0.224, 0.225]

class MaskNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bilinear=True):
        super(MaskNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bilinear = use_bilinear

        if self.use_bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = nn.Conv2d(
                self.in_channels, self.out_channels, 3, stride=1, padding=1)
            self.conv1_norm = nn.BatchNorm2d(self.out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                self.in_channels, self.out_channels, 2, stride=2, padding=0)
            self.conv1 = nn.Conv2d(
                self.out_channels, self.out_channels, 3, stride=1, padding=1)
            self.conv1_norm = nn.BatchNorm2d(self.out_channels)

        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, stride=1, padding=1)
        self.conv2_norm = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.up(x)
        x = self.relu(self.conv1_norm(self.conv1(x)))
        x = self.relu(self.conv2_norm(self.conv2(x)))
        return x

class MaskNet(nn.Module):
    def __init__(self, in_channels):
        super(MaskNet, self).__init__()
        self.in_channels = in_channels

        self.block1 = MaskNetBlock(self.in_channels, 256, use_bilinear=True)
        self.block2 = MaskNetBlock(256, 128, use_bilinear=True)
        # self.block3 = MaskNetBlock(128, 64, use_bilinear=True)
        self.conv_last = nn.Conv2d(128, 1, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x, target_masks=None):
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        x = self.conv_last(x)

        # device = x.device
        if target_masks is not None:
            losses = {
                'loss_mask': nn.BCEWithLogitsLoss()(x, target_masks)
            }
            return x, losses
        else:
            return x

class UnetMask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetMask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.relu = nn.ReLU()

    def forward(self, x, target_masks=None):
        x = self.conv(x)

        if target_masks is not None:
            losses = {
                'loss_mask': nn.BCEWithLogitsLoss()(x, target_masks)
            }
            return x, losses
        else:
            return x


class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, mask_net, transform, input_img_num=6, depth_estimator_model_path='_depth_net.pth'):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        # self.backbone_list = nn.ModuleList([b for b in backbone_list])
        # self.backbone_num = len(backbone_list)
        self.backbone = UNet(4, 1) # backbone
        self.backbone_ = UNet(4, 64)
        # self.backbone_ = resnet_fpn_backbone('resnet18', False)
        self.input_img_num = input_img_num
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.mask_net = UnetMask(6, 1) # mask_net
        self.backbone_out_channels = 64 # backbone.out_channels # backbone_list[0].out_channels

        self.depth_estimator_model_path = depth_estimator_model_path
        self.depth_estimator = VggDepthEstimator()
        self.depth_estimator.load_state_dict(torch.load(self.depth_estimator_model_path))
        self.depth_resize = nn.Upsample(size=(400, 400), mode='bilinear', align_corners=True)

        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.target_transform = GeneralizedRCNNTransform(
            400, 400, [0., 0., 0.], [1., 1., 1.])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor()
        ])

        self.depth_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 416)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def forward(self, _images, _targets=None, return_result=False, return_losses=False):
        # assert images.size(1) == self.backbone_num
        bs = _images.size(0)
        assert bs == 1

        device = _images.device

        # Process images
        images = torch.zeros(1, 6, 3, 400, 400)
        depths = torch.zeros(1, 6, 3, 128, 416)
        for i in range(6):
            images[0, i] = self.img_transform(_images[0, i].cpu())
            depths[0, i] = self.depth_transform(_images[0, i].cpu())
        del _images
        images = images.to(device)
        depths = depths.to(device)

        # Get depth map
        depths = self.depth_estimator(depths.squeeze(0))[0]
        depths = self.depth_resize(depths.unsqueeze(1))
        depths = depths.view(1, 6, 1, 400, 400)
        images = torch.cat((images, depths), dim=2)
        del depths

        # Process targets
        dis = torch.mean(_targets[0]['boxes'], dim=2) - torch.tensor([400., 400.])
        index_1 = torch.sqrt(torch.sum(torch.pow(dis, 2), dim=1)) < 300.
        index_2 = (_targets[0]['labels'] == 0) | (_targets[0]['labels'] == 2) |\
            (_targets[0]['labels'] == 4) | (_targets[0]['labels'] == 5)
        label_index = index_1 * index_2

        targets = [copy.deepcopy(_targets[0])]
        targets[0]['boxes'] = targets[0]['boxes'][label_index]
        targets[0]['labels'] = targets[0]['labels'][label_index]

        targets = [{k: v for k, v in t.items()} for t in targets]
        # targets[0]['old_boxes'] = targets[0]['boxes'] / 2.
        min_coordinates, _ = torch.min(targets[0]['boxes'], 2)
        max_coordinates, _ = torch.max(targets[0]['boxes'], 2)
        targets[0]['boxes'] = torch.cat([min_coordinates, max_coordinates], 1)
        temp_tensor = torch.zeros(1, 3, 800, 800)
        _, targets = self.target_transform(temp_tensor, targets)

        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # images, targets = self.transform(images, targets)
        device = images.device
        images = ImageList(images, ((400, 400),) * images.size(0))
        target_masks = torch.stack([t['masks'].float().to(device) for t in targets])
        targets = [{k: v.to(device) for k, v in t.items() if k != 'masks'} for t in targets]

        # Mask backbone
        features_list = torch.stack(
            [self.backbone(images.tensors[:, i]) for i in range(self.input_img_num)], dim=1)

        feature_h, feature_w = features_list.size()[-2:]
        combined_feature_map = features_list.view(
            bs, self.input_img_num, feature_h, feature_w)

        masks, mask_losses = self.mask_net(combined_feature_map, target_masks)

        del features_list # , combined_feature_map
        torch.cuda.empty_cache()

        # Detction backbone
        features_list = torch.stack(
            [self.backbone_(images.tensors[:, i]) for i in range(self.input_img_num)], dim=1)

        feature_h, feature_w = features_list.size()[-2:]
        detection_combined_feature_map = features_list.view(
            bs, 64 * self.input_img_num, 400, 400)
        del features_list
        torch.cuda.empty_cache()

        road_map_features = OrderedDict([('0', combined_feature_map)])
        detection_features = OrderedDict([('0', detection_combined_feature_map)])

        proposals, proposal_losses = self.rpn(images, road_map_features, targets)
        # proposals, proposal_losses = self.rpn(images, detection_features, targets)
        # try:
        #     detections, detector_losses = self.roi_heads(detection_features, proposals, images.image_sizes, targets)
        #     detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        # except RuntimeError as e:
        #     print(e)
        #     detections = None
        #     detector_losses = {
        #         'loss_box_reg': torch.zeros(1),
        #         'loss_classifier': torch.zeros(1)}
        detections, detector_losses = self.roi_heads(detection_features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(mask_losses)

#         mask_ts = 0.
#         mask_ts_numerator = 0
#         mask_ts_denominator = 1
#         detection_ts = 0.
#         detection_ts_numerator = 0
#         detection_ts_denominator = 1

        if return_result:
            with torch.no_grad():
                # Get mask threat score
                _masks = self.mask_transform(masks.cpu().squeeze(0)).unsqueeze(0)
                mask_ts, mask_ts_numerator, mask_ts_denominator = get_mask_threat_score(
                    # masks.cpu(), target_masks.cpu())
                    _masks, _targets[0]['masks'].float())

                # Get object detection threat score
                detection_ts_numerator = 0
                detection_ts_denominator = 0
                if detections is not None:
                    for i in range(len(detections)):
                        if len(detections[i]['boxes']) == 0:
                            continue
                        min_coordinates, _ = torch.min(_targets[0]['boxes'], 2)
                        max_coordinates, _ = torch.max(_targets[0]['boxes'], 2)
                        _targets[0]['boxes'] = torch.cat([min_coordinates, max_coordinates], 1)
                        _detection = detections[i]['boxes'].cpu().view(-1, 2, 2) * 2
                        _, d_ts_n, d_ts_d = compute_ats_bounding_boxes(
                            # detections[i]['boxes'].cpu().view(-1, 2, 2),
                            _detection,
                            _targets[i]['boxes'].cpu().view(-1, 2, 2))
                        detection_ts_numerator += d_ts_n
                        detection_ts_denominator += d_ts_d
                    try:
                        detection_ts = detection_ts_numerator / detection_ts_denominator
                    except ZeroDivisionError:
                        detection_ts = 0.
                else:
                    detection_ts = 0.
    #             detection_ts, detection_ts_numerator, detection_ts_denominator =\
    #                 get_detection_threat_score(cpu_detections, targets, 0.5)
            return mask_ts, mask_ts_numerator,\
                   mask_ts_denominator, detection_ts, detection_ts_numerator,\
                   detection_ts_denominator, detections, masks,
        else:
            return losses

class ModifiedFasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
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
                 box_score_thresh=0.05, box_nms_thresh=0.1, box_detections_per_img=100,
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

        out_channels = 6 # backbone.out_channels

        if rpn_anchor_generator is None:
            # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            # anchor_sizes = ((4,), (8,), (16,),)
            anchor_sizes = ((16,), (32,), (64,),)
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
                64 * resolution ** 2, # out_channels * resolution ** 2,
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

        super(ModifiedFasterRCNN, self).__init__(backbone, rpn, roi_heads, mask_net, transform)


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
