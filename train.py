import os
import time
import pickle
import random
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from models import get_model
from datasets import get_loader

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
    default='adam',
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
    '--batch_size',
    type=int,
    default=2,
    help='batch size of dataloader')
parser.add_argument(
    '--backprop_batch_size',
    type=int,
    default=16,
    help='real batch size')
parser.add_argument(
    '--num_workers',
    type=int,
    default=0,
    help='num_workers in dataloader')
parser.add_argument(
    '--model',
    type=str,
    default='new_faster_rcnn',
    help='which model to use')
parser.add_argument(
    '--dataset',
    type=str,
    default='new_faster_rcnn',
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

args = parser.parse_args()
assert args.optimizer in ['sgd', 'adam']
assert args.model in ['basic', 'faster_rcnn', 'new_faster_rcnn']
assert args.dataset in ['basic', 'faster_rcnn', 'new_faster_rcnn']

def train(epoch):
    global all_train_masks
    model.train()

    t = time.time()
    loader_len = len(train_dataloader)

    total_loss = 0.
    total_rpn_box_reg_loss = 0.
    total_rpn_cls_loss = 0.
    total_roi_box_reg_loss = 0.
    total_roi_cls_loss = 0.
    total_mask_loss = 0.
    total_mask_ts_numerator = 0
    total_mask_ts_denominator = 0
    total_detection_ts_numerator = 0
    total_detection_ts_denominator = 0
    batch_count = 0
    for batch_idx, data in enumerate(train_dataloader):
        imgs = torch.stack(data[0]).to(device)
        targets = data[1]
        results = model(imgs, targets, return_result=True)
        losses = results[0]
        mask_ts, mask_ts_numerator, mask_ts_denominator = results[1:4]
        detection_ts, detection_ts_numerator, detection_ts_denominator = results[4:7]

        loss = sum(l for l in losses.values())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        batch_count += len(imgs)
        if batch_count >= args.backprop_batch_size or batch_idx == loader_len - 1:
            optimizer.step()
            optimizer.zero_grad()
            batch_count = 0

        # evaluate
        with torch.no_grad():
            total_loss += loss.cpu().item()
            total_rpn_box_reg_loss += losses['loss_rpn_box_reg'].item()
            total_rpn_cls_loss += losses['loss_objectness'].item()
            total_roi_box_reg_loss += losses['loss_box_reg'].item()
            total_roi_cls_loss += losses['loss_classifier'].item()
            total_mask_loss += losses['loss_mask'].item()
            total_mask_ts_numerator += mask_ts_numerator
            total_mask_ts_denominator += mask_ts_denominator
            total_detection_ts_numerator += detection_ts_numerator
            total_detection_ts_denominator += detection_ts_denominator

            # DEBUG
            all_train_masks.append(results[8].cpu().detach())

        # Log
        print(('Train Epoch {} {}/{} ({:.3f}s) | '
               'loss: {:.3f} | '
               'rpn box regression loss: {:.3f} | '
               'rpn classifier loss: {:.3f} | '
               'roi box regression loss: {:.3f} | '
               'roi classifier loss: {:.3f} | '
               'mask loss: {:.3f} | '
               'mask threat score: {:.3f} | '
               'detection threat score: {:.3f}').format(
            epoch+1, batch_idx+1, loader_len, time.time() - t,
            loss.item(), losses['loss_rpn_box_reg'].item(),
            losses['loss_objectness'].item(), losses['loss_box_reg'].item(),
            losses['loss_classifier'].item(), losses['loss_mask'].item(),
            mask_ts, detection_ts))
        t = time.time()

    try:
        total_mask_ts = total_mask_ts_numerator / total_mask_ts_denominator
    except ZeroDivisionError:
        total_mask_ts = 0.

    try:
        total_detection_ts = total_detection_ts_numerator / total_detection_ts_denominator
    except ZeroDivisionError:
        total_detection_ts = 0.

    total_results = {
        'loss': total_loss / loader_len,
        'rpn_box_reg_loss': total_rpn_box_reg_loss / loader_len,
        'rpn_cls_loss': total_rpn_cls_loss / loader_len,
        'roi_box_reg_loss': total_roi_box_reg_loss / loader_len,
        'roi_cls_loss': total_roi_cls_loss / loader_len,
        'mask_loss': total_mask_loss / loader_len,
        'mask_ts': total_mask_ts,
        'detection_ts': total_detection_ts
    }
    return total_results

def validate(epoch):
    global all_val_masks
    with torch.no_grad():
        # model.eval()

        t = time.time()
        loader_len = len(val_dataloader)

        total_loss = 0.
        total_rpn_box_reg_loss = 0.
        total_rpn_cls_loss = 0.
        total_roi_box_reg_loss = 0.
        total_roi_cls_loss = 0.
        total_mask_loss = 0.
        total_mask_ts_numerator = 0
        total_mask_ts_denominator = 0
        total_detection_ts_numerator = 0
        total_detection_ts_denominator = 0
        for batch_idx, data in enumerate(val_dataloader):
            imgs = torch.stack(data[0]).to(device)
            targets = data[1]
            results = model(imgs, targets, return_result=True)
            losses = results[0]
            mask_ts, mask_ts_numerator, mask_ts_denominator = results[1:4]
            detection_ts, detection_ts_numerator, detection_ts_denominator = results[4:7]

            loss = sum(l for l in losses.values())

            # DEBUG
            all_val_masks.append(results[8].cpu().detach())

            total_loss += loss.cpu().item()
            total_rpn_box_reg_loss += losses['loss_rpn_box_reg'].item()
            total_rpn_cls_loss += losses['loss_objectness'].item()
            total_roi_box_reg_loss += losses['loss_box_reg'].item()
            total_roi_cls_loss += losses['loss_classifier'].item()
            total_mask_loss += losses['loss_mask'].item()
            total_mask_ts_numerator += mask_ts_numerator
            total_mask_ts_denominator += mask_ts_denominator
            total_detection_ts_numerator += detection_ts_numerator
            total_detection_ts_denominator += detection_ts_denominator

            # Log
            print(('Val Epoch {} {}/{} ({:.3f}s) | '
               'loss: {:.3f} | '
               'rpn box regression loss: {:.3f} | '
               'rpn classifier loss: {:.3f} | '
               'roi box regression loss: {:.3f} | '
               'roi classifier loss: {:.3f} | '
               'mask loss: {:.3f} | '
               'mask threat score: {:.3f} | '
               'detection threat score: {:.3f}').format(
            epoch+1, batch_idx+1, len(val_dataloader), time.time() - t,
            loss.item(), losses['loss_rpn_box_reg'].item(),
            losses['loss_objectness'].item(), losses['loss_box_reg'].item(),
            losses['loss_classifier'].item(), losses['loss_mask'],
            mask_ts, detection_ts))
            t = time.time()

    try:
        total_mask_ts = total_mask_ts_numerator / total_mask_ts_denominator
    except ZeroDivisionError:
        total_mask_ts = 0.

    try:
        total_detection_ts = total_detection_ts_numerator / total_detection_ts_denominator
    except ZeroDivisionError:
        total_detection_ts = 0.

    total_results = {
        'loss': total_loss / loader_len,
        'rpn_box_reg_loss': total_rpn_box_reg_loss / loader_len,
        'rpn_cls_loss': total_rpn_cls_loss / loader_len,
        'roi_box_reg_loss': total_roi_box_reg_loss / loader_len,
        'roi_cls_loss': total_roi_cls_loss / loader_len,
        'mask_loss': total_mask_loss / loader_len,
        'mask_ts': total_mask_ts,
        'detection_ts': total_detection_ts
    }
    return total_results

if __name__ == '__main__':
    # DEBUG
    all_train_masks = []
    all_val_masks = []

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
    device_count = torch.cuda.device_count()
    if device_count > 1:
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
            datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        store_dir = os.path.join(args.results_dir, store_name)
        os.makedirs(store_dir)

    results = {
        'args': args,
        'train_loss': [],
        'train_rpn_box_reg_loss': [],
        'train_rpn_cls_loss': [],
        'train_roi_box_reg_loss': [],
        'train_roi_cls_loss': [],
        'train_mask_loss': [],
        'train_mask_ts': [],
        'train_detection_ts': [],
        'val_loss': [],
        'val_rpn_box_reg_loss': [],
        'val_rpn_cls_loss': [],
        'val_roi_box_reg_loss': [],
        'val_roi_cls_loss': [],
        'val_mask_loss': [],
        'val_mask_ts': [],
        'val_detection_ts': [],
    }
    if not args.debug:
        store_file = f'{args.dataset}_dataset_{args.model}_model'
        store_file = os.path.join(store_dir, store_file)

    print("Dataset: {} | Model: {}\n".format(args.dataset, args.model))

    # Training and validation
    last_val_loss = 1e8
    last_saved_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('Starting Epoch {}\n'.format(epoch))

        # Train
        # train_loss, train_mask_ts, train_detection_ts = train(epoch)
        train_results = train(epoch)
        results['train_loss'].append(train_results['loss'])
        results['train_rpn_box_reg_loss'].append(train_results['rpn_box_reg_loss'])
        results['train_rpn_cls_loss'].append(train_results['rpn_cls_loss'])
        results['train_roi_box_reg_loss'].append(train_results['roi_box_reg_loss'])
        results['train_roi_cls_loss'].append(train_results['roi_cls_loss'])
        results['train_mask_loss'].append(train_results['mask_loss'])
        results['train_mask_ts'].append(train_results['mask_ts'])
        results['train_detection_ts'].append(train_results['detection_ts'])
        print(('\nTotal train loss: {:.3f} | '
               'train rpn box regression loss: {:.3f} | '
               'train rpn classifier loss: {:.3f} | '
               'train roi box regression loss: {:.3f} | '
               'train roi classifier loss: {:.3f} | '
               'train mask loss: {:.3f} | '
               'train mask ts: {:.3f} | '
               'train detection ts: {:.3f}\n').format(
            train_results['loss'], train_results['rpn_box_reg_loss'],
            train_results['rpn_cls_loss'], train_results['roi_box_reg_loss'],
            train_results['roi_cls_loss'], train_results['mask_loss'],
            train_results['mask_ts'],
            train_results['detection_ts']))

        if scheduler:
            scheduler.step()

        # Val
        val_results = validate(epoch)
        results['val_loss'].append(val_results['loss'])
        results['val_rpn_box_reg_loss'].append(val_results['rpn_box_reg_loss'])
        results['val_rpn_cls_loss'].append(val_results['rpn_cls_loss'])
        results['val_roi_box_reg_loss'].append(val_results['roi_box_reg_loss'])
        results['val_roi_cls_loss'].append(val_results['roi_cls_loss'])
        results['val_mask_loss'].append(val_results['mask_loss'])
        results['val_mask_ts'].append(val_results['mask_ts'])
        results['val_detection_ts'].append(val_results['detection_ts'])
        print(('\nTotal val loss: {:.3f} | '
               'val rpn box regression loss: {:.3f} | '
               'val rpn classifier loss: {:.3f} | '
               'val roi box regression loss: {:.3f} | '
               'val roi classifier loss: {:.3f} | '
               'val mask loss: {:.3f} | '
               'val mask ts: {:.3f} | '
               'val detection ts: {:.3f}\n').format(
            val_results['loss'], val_results['rpn_box_reg_loss'],
            val_results['rpn_cls_loss'], val_results['roi_box_reg_loss'],
            val_results['roi_cls_loss'], val_results['mask_loss'],
            val_results['mask_ts'], val_results['detection_ts']))

        if not args.debug: #  and last_val_loss > val_results['loss']:
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': val_results['loss']
            }
            print('\n***\nSaving epoch {}\n***\n'.format(epoch+1))
            torch.save(state, os.path.join(store_dir, f'epoch{epoch}.pth'))
            last_val_loss = val_results['loss']
            last_saved_epoch = epoch

    # Save results to a file
    if not args.debug:
        with open(store_file, 'wb') as f:
            pickle.dump(results, f)
