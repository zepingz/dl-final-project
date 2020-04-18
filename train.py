import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import get_model
from datasets import get_loader
from utils import collate_fn, draw_box, get_threat_score

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root',
    type=str,
    default='../shared/dl/data',
    help='data dirctory')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='how many epochs')
parser.add_argument(
    '--optimizer',
    type=str,
    default='sgd',
    help='which optimizer to use')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    help='learning rate')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='sgd momentum')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='weight decay')
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='random seed')
parser.add_argument(
    '--result_dir',
    type=str,
    default='./result',
    help='directory to store result and model')
parser.add_argument(
    '--batch_size',
    type=int,
    default=4,
    help='batch size')
parser.add_argument(
    '--num_workers',
    type=int,
    default=0,
    help='num_workers in dataloader')
parser.add_argument(
    '--model',
    type=str,
    default='basic',
    help='which model to use')
parser.add_argument(
    '--dataset',
    type=str,
    default='original',
    help='which dataloader to use')
parser.add_argument(
    '--results_dir',
    type=str,
    default='./results',
    help='where to store the results')
parser.add_argument(
    '--store_name',
    type=str,
    default='',
    help='name of store directory')
parser.add_argument(
    '--debug',
    default=False,
    action='store_true',
    help='debug mode (without saving to a directory)')
parser.add_argument(
    '--resume_dir',
    type=str,
    default='',
    help='dir where we resume from checkpoint')
parser.add_argument(
    '--criterion',
    type=str,
    default='', 
    help='loss function')
parser.add_argument(
    '--use_scheduler',
    default=False,
    action='store_true',
    help='whether to use the scheduler or not')
parser.add_argument(
    '--schedule_milestones',
    type=str,
    default='150,250',
    help='the milestones in the LR')
parser.add_argument(
    '--schedule_gamma',
    type=float,
    default=0.1,
    help='the default LR gamma')

assert args.optimizer in ['sgd', 'adam']
assert args.model in ['basic']
assert args.dataset in ['original']

# TODO: move this to a dataloader
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

def train(epoch):
    total_train_loss = 0.
    total_train_ts = 0.
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
        
        total_train_loss += loss.detach().cpu().item()
        total_train_ts += ts # TODO: this might be wrong

        # Log
        if (batch_idx + 1) % 1 == 0:
            print('Train Epoch {} {}/{} | loss: {} | threat score: {:.3f}'.format(
                epoch+1, batch_idx+1, len(train_dataloader),
                loss.item(), ts), end='\r')
            
    return total_train_loss, total_train_ts

def validate(epoch):
    for batch_idx, data in enumerate(val_dataloader):
        output = model(imgs)
        loss = criterion(output, road_imgs)
        if batch_idx % 1:
            print('Val Epoch {} {}/{} | loss: {}'.format(
                epoch, batch_idx, 1, loss.item()))
            
if __name__ == '__main__':
    # Set up random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        # Setting bechmark to False might slow down the training speed
        torch.backends.cudnn.benchmark = True # False
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    start_epoch = 0
    
    # Set up data
    # TODO: compute the actual mean and std

    # TODO: make a new dataset class
    print("Loading data")
    train_dataloader = get_loader(args)

    # Set up model and loss function
    print("Creating model")
    model, criterion = get_model(args)
    model = model.to(device)
    
    if args.resume_dir and not args.debug:
        # Load checkpoint
        print('==> Resuming from checkpoint')
        # TODO: change ckpt.pth
        checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    
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
        
    # Set up scheduler
    if args.use_scheduler:
        milestones = [int(k) for k in args.schedule_milestones.split(',')]
        scheduler = optim.lr_schedulre.MultiSteepLR(
            optimizer, milestones=milestones, gamma=args.schedule_gamma)
    else:
        scheduler = None
        
    # Set up results directory
    if not args.debug:
        store_name = args.store_name if args.store_name else\
            datetim.today().strftime('%Y-%m-%d-%H-%M-%S')
        store_dir = os.path.join(args.results_dir, store_name)
        os.makedirs(store_dir)
        
    results = {
        'args': args,
        'train_ts': [],
        'train_loss': [],
        # 'val_ts': [],
        # 'val_loss': []
    }
    if not args.debug:
        store_file = '{}_dataset_{}_model'.format(
            args.dataset, args.model)
        store_file = os.path.join(store_dir, store_file)
        
    print("Dataset: {} | Model: {}\n".format(args.dataset, args.model))

    # Training and validation
    last_val_loss = 1e8
    last_saved_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('Starting Epoch {}'.format(epoch))
        
        train_loss, train_ts = train(epoch)
        results['train_loss'].append(train_loss)
        results['train_ts'].append(train_ts)
        print('Train loss: {} | Train ts: {}'.format(
            train_loss, train_ts))
        
        if scheduler:
            scheduler.step()

        val_loss = 0
        # val_loss, val_ts = validate(epoch)
        # results['val_loss'].append(val_loss)
        # results['val_ts'].append(val_ts)
    
        if not args.debug and epoch >= last_saved_epoch + 10 and\
            last_val_loss > val_loss:
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                # 'step': step,
                'loss': train_loss # val_loss
            }
            print('\n***\nSaving epoch {}\n***\n'.format(epoch))
            torch.save(
                state, os.path.join(store_dir, 'epoch{}.pth'.format(epoch)))
            last_val_loss = val_loss
            last_saved_epoch = epoch
        
    # Save results to a file
    if not args.debug:
        with open(store_file, 'wb') as f:
            pickle.dump(results, f)