import os
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,\
    MaskRCNNHeads

from models.basic import BasicModel
from utils import collate_fn, draw_box, get_threat_score
from datasets.original_dataset import UnlabeledDataset, LabeledDataset

# TODO: add resume dir
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root', type=str, default='../shared/dl/data', help='data dirctory')
parser.add_argument(
    '--epochs', type=int, default=100, help='how many epochs')
parser.add_argument(
    '--optimizer', type=str, default='sgd', help='which optimizer to use')
parser.add_argument(
    '--lr', type=float, default=0.01, help='learning rate')
parser.add_argument(
    '--momentum', type=float, default=0.9, help='sgd momentum')
parser.add_argument(
    '--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument(
    '--seed', type=int, default=0, help='random seed')
parser.add_argument(
    '--result_dir', type=str, default='./result', help='directory to store result and model')
parser.add_argument(
    '--batch_size', type=int, default=4, help='batch size')
parser.add_argument(
    '--num_workers', type=int, default=0, help='num_workers in dataloader')
parser.add_argument(
    '--model', type=str, default='basic', help='which model to use')
parser.add_argument(
    '--dataset', type=str, default='original', help='which dataloader to use')
args = parser.parse_args()

assert args.optimizer in ['sgd', 'adam']
assert args.model in ['basic']
assert args.dataset in ['original']

# Set up random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    # Setting bechmark to False might slow down the training speed
    torch.backends.cudnn.benchmark = True # False

unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(epoch):
    for batch_idx, data in enumerate(train_dataloader):
        imgs, _, road_imgs, _ = data
        imgs = torch.stack(imgs).to(device)
        road_imgs = torch.stack(road_imgs) # .to(device)
        road_imgs_target = torch.stack([
            road_img_transform(road_img.float()) for road_img in road_imgs]).to(device)

        output = model(imgs)
        loss = criterion(output, road_imgs_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate
        # TODO: use average instead
        pred_road_imgs = torch.stack([resize_transform(
            (nn.Sigmoid()(pred_img) > 0.5).int()) for pred_img in output.cpu()])

        ts = get_threat_score(pred_road_imgs, road_imgs.float())

        # Log
        if (batch_idx + 1) % 1 == 0:
            print('Train Epoch {} {}/{} | loss: {} | threat score: {:.3f}'.format(
                epoch+1, batch_idx+1, len(train_dataloader),
                loss.item(), ts), end='\r')

def validate(epoch):
    for batch_idx, data in enumerate(val_dataloader):
        output = model(imgs)
        loss = criterion(output, road_imgs)
        if batch_idx % 1:
            print('Val Epoch {} {}/{} | loss: {}'.format(
                epoch, batch_idx, 1, loss.item()))

if __name__ == '__main__':
    # Set up data
    # TODO: compute the actual mean and std
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

    # TODO: make a new dataset class
    print("Loading data")
    if args.dataset == 'original':
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

    # Set up model and loss function
    print("Creating model")
    if args.model == 'basic':
        model = BasicModel()
        criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    
    # Set up optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train(epoch)
        # validate(epoch)
    
    # TODO: more detailed saving (also save result)
    # torch.save(model.state_dict(), args.result_dir)