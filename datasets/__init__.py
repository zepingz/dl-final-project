import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from .basic_dataset import BasicLabelledDataset
from .faster_rcnn_dataset import FasterRCNNLabelledDataset
from .new_faster_rcnn_dataset import NewFasterRCNNLabelledDataset
from utils.data import collate_fn
from utils.target_transforms import TargetResize

MEAN = [0.5459, 0.5968, 0.6303] # [0.485, 0.456, 0.406]
STD = [0.3178, 0.3246, 0.3278] # [0.229, 0.224, 0.225]

unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)

def get_loader(args):
    if args.dataset == 'basic':
        img_transform = transforms.Compose([
            # transforms.Resize(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        road_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 928)),
            transforms.ToTensor(),
        ])
        dataset = BasicLabelledDataset(
            args.data_root,
            img_transform=img_transform,
            road_transform=road_transform)

        # Make it hard by hiding some full scenes
        split_num = int(len(labeled_scene_index) * 0.8) * 126
        train_indices = range(split_num)
        val_indices = range(split_num, len(dataset))
        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)

        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)
        val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)
    elif args.dataset == 'faster_rcnn':
        img_transform = transforms.Compose([
            # transforms.Resize(),
            transforms.ToTensor(),
            # transforms.Normalize(MEAN, STD)
        ])
        target_transform = TargetResize(800, (512, 920))
        dataset = FasterRCNNLabelledDataset(
            args.data_root,
            img_transform=img_transform,
            target_transform=target_transform)

        # Make it hard by hiding some full scenes
        split_num = int(len(labeled_scene_index) * 0.8) * 126
        train_indices = range(split_num)
        val_indices = range(split_num, len(dataset))
        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)

        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)
    elif args.dataset == 'new_faster_rcnn':
        img_transform = transforms.Compose([
            # transforms.Resize((800, 800)),
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        dataset = NewFasterRCNNLabelledDataset(args.data_root, img_transform)

        # Make it hard by hiding some full scenes
        split_num = int(len(labeled_scene_index) * 0.8) * 126
        train_indices = range(split_num)
        val_indices = range(split_num, len(dataset))

        # DEBUG
        train_indices = list(train_indices) # [::4]
        val_indices = list(val_indices) # [::4]
        # train_indices = np.random.choice(train_indices, 4, replace=False)
        # val_indices = np.random.choice(val_indices, 1, replace=False)

        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)

        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)

    return train_dataloader, val_dataloader
