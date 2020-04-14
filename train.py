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

# TODO: add argparse stuff
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root', type=str, default='../data', help='data dirctory')
parser.add_argument(
    '--epochs', type=int, default=100, help='how many epochs')
parser.add_argument(
    '--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

# Set up random seed
# TODO: set up gpu random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

annotation_csv = os.path.join(args.data_root, 'annotation.csv')
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
        # TODO: set up threat score
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
    labeled_set = LabeledDataset(
        image_folder=args.data_root,
        annotation_file=annotation_csv,
        scene_index=labeled_scene_index,
        transform=transform,
        extra_info=True)
    train_set = torch.utils.data.Subset(
        labeled_trainset, np.random.choice(range(len(labeled_set)), 8))
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)

    # Set up model, optimizer and loss function
    model = BasicModel()
    model = model.to(device)
    # TODO: add optimizer args
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        train(epoch)
        # validate(epoch)
        # TODO: save model
