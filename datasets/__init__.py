import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from datasets.original_dataset import LabeledDataset
from utils import collate_fn

unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)

def get_loader(args):
    if args.dataset == 'original':
        transform = transforms.Compose([
            # transforms.Resize(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        road_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 928)),
            transforms.ToTensor(),
        ])
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor()
        ])
        labeled_set = LabeledDataset(
            image_folder=args.data_root,
            annotation_file=os.path.join(args.data_root, 'annotation.csv'),
            scene_index=labeled_scene_index,
            transform=transform,
            extra_info=True)
        train_set = torch.utils.data.Subset(
            labeled_set, np.random.choice(range(len(labeled_set)), 8))
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)
    else:
        raise 'No such dataloader'
        
    return train_dataloader