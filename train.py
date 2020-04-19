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
from utils.data import collate_fn
from utils.target_transforms import TargetResize
from utils.visualize import draw_box, visualize_target
# from utils.evaluate import get_mask_threat_score, get_detection_threat_score

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
    default=0.001,
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
    default=8,
    help='batch size')
parser.add_argument(
    '--num_workers',
    type=int,
    default=0,
    help='num_workers in dataloader')
parser.add_argument(
    '--model',
    type=str,
    default='faster_rcnn',
    help='which model to use')
parser.add_argument(
    '--dataset',
    type=str,
    default='faster_rcnn',
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
assert args.model in ['basic', 'faster_rcnn']
assert args.dataset in ['basic', 'faster_rcnn']


def train(epoch):
    model.train()
    
    total_loss = 0.
    total_mask_ts_numerator = 0
    total_mask_ts_denominator = 0
    total_detection_ts_numerator = 0
    total_detection_ts_denominator = 0
    for batch_idx, data in enumerate(train_dataloader):
        results = model.get_loss(data, device)
        loss = results[0]
        mask_ts, mask_ts_numerator, mask_ts_denominator = results[1:4]
        detection_ts, detection_ts_numerator, detection_ts_denominator = results[4:]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            total_loss += loss.cpu().item()
            total_mask_ts_numerator += mask_ts_numerator
            total_mask_ts_denominator += mask_ts_denominator
            total_detection_ts_numerator += detection_ts_numerator
            total_detection_ts_denominator += detection_ts_denominator

        # Log
        print(('Train Epoch {} {}/{} | loss: {:.3f} | '
               'mask threat score: {:.3f} |'
               'detection threat score: {:.3f}').format(
            epoch+1, batch_idx+1, len(train_dataloader),
            loss.item(), mask_ts, detection_ts), end='\r')
            
    total_loss /= len(train_dataloader.dataset)
    total_mask_ts = total_mask_ts_numerator / total_mask_ts_denominator
    total_detection_ts = total_detection_ts_numerator / total_detection_ts_denominator
    return total_loss, total_mask_ts, total_detection_ts

def validate(epoch):
    with torch.no_grad():
        model.eval()

        total_loss = 0.
        total_mask_ts_numerator = 0
        total_mask_ts_denominator = 0
        total_detection_ts_numerator = 0
        total_detection_ts_denominator = 0
        for batch_idx, data in enumerate(val_dataloader):
            results = model.get_loss(data, device)
            loss = results[0]
            mask_ts, mask_ts_numerator, mask_ts_denominator = results[1:4]
            detection_ts, detection_ts_numeeator, detection_ts_denominator = results[4:]

            total_loss += loss.cpu().item()
            total_mask_ts_numerator += mask_ts_numerator
            total_mask_ts_denominator += mask_ts_denominator
            total_detection_ts_numerator += detection_ts_numerator
            total_detection_ts_denominator += detection_ts_denominator

            # Log
            print('Val Epoch {} {}/{} | loss: {:.3f} | '
                  'mask thraet score: {:.3f} | '
                  'detection threat score: {:.3f}'.format(
                epoch+1, batch_idx+1, len(val_dataloader),
                loss.item(), mask_ts, detection_ts), end='\r')

        total_loss /= len(val_dataloader.dataset)
        total_mask_ts = total_mask_ts_numerator / total_mask_ts_denominator
        total_detection_ts = total_detection_ts_numerator / total_detection_ts_denominator
    return total_loss, total_mask_ts, total_detection_ts
            
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
    print("Loading data")
    train_dataloader, val_dataloader = get_loader(args)

    # Set up model and loss function
    print("Creating model")
    model = get_model(args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    if args.resume_dir and not args.debug:
        # Load checkpoint
        print('==> Resuming from checkpoint')
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
        'train_loss': [],
        'train_mask_ts': [],
        'train_detection_ts': [],
        'val_loss': [],
        'val_mask_ts': [],
        'val_detection_ts': [],
    }
    if not args.debug:
        store_file = f'{args.datset}_dataset_{args.model}_model'
        store_file = os.path.join(store_dir, store_file)
        
    print("Dataset: {} | Model: {}\n".format(args.dataset, args.model))

    # Training and validation
    last_val_loss = 1e8
    last_saved_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('Starting Epoch {}'.format(epoch))
        
        train_loss, train_mask_ts, train_detection_ts = train(epoch)
        results['train_loss'].append(train_loss)
        results['train_mask_ts'].append(train_mask_ts)
        results['train_detection_ts'].append(train_detection_ts)
        print(('\nTotal train loss: {:.3f} | train mask ts: {:.3f} | '
               'train detection ts: {:.3f}').format(
            train_loss, train_mask_ts, train_detection_ts))
        
        if scheduler:
            scheduler.step()

        val_loss = 0
        val_loss, val_mask_ts, val_detection_ts = validate(epoch)
        results['val_loss'].append(val_loss)
        results['val_mask_ts'].append(val_mask_ts)
        results['val_detection_ts'].append(val_detection_ts)
        print(('\nTotal val loss: {:.3f} | val mask ts: {:.3f} | '
               'val detection ts: {:.3f}').format(
            val_loss, val_mask_ts, val_detection_ts))
    
        if not args.debug and last_val_loss > val_loss:
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': val_loss
            }
            print('\n***\nSaving epoch {}\n***\n'.format(epoch))
            torch.save(state, os.path.join(store_dir, f'epoch{epoch}.pth'))
            last_val_loss = val_loss
            last_saved_epoch = epoch
        
    # Save results to a file
    if not args.debug:
        with open(store_file, 'wb') as f:
            pickle.dump(results, f)