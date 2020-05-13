import os
import time
import copy
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

from helper import compute_ts_road_map, compute_ats_bounding_boxes
from torchvision.models.detection.transform import GeneralizedRCNNTransform
target_transform = GeneralizedRCNNTransform(
        800, 800, [0., 0., 0.], [1., 1., 1.])

def get_mask_ts(mask, target):
    mask = nn.Sigmoid()(mask) > 0.5

    temp_tensor = torch.zeros(1, 3, 400, 400)
    temp_target = [{'masks': mask,
                    'boxes': torch.tensor([[1., 1., 1., 1.]])}]
    _, temp_target = target_transform(temp_tensor, temp_target)
    predicted_road_map = temp_target[0]['masks'][0, :1]
    
    ts_road_map = compute_ts_road_map(predicted_road_map, target[0]['masks'])
    return ts_road_map

def get_detection_ts(detection, target):
    detection = detection[0]
    true_index = detection['scores'] > 0.
    detection = detection['boxes'][true_index].cpu()

    if len(detection) == 0:
        return 0.

    pred_detection = []
    for i in range(len(detection)):
        min_x = detection[i][0].item()
        min_y = detection[i][1].item()
        max_x = detection[i][2].item()
        max_y = detection[i][3].item()

        pred_box = torch.tensor([
            [max_x, max_x, min_x, min_x],
            [max_y, min_y, max_y, min_y]])
        pred_detection.append(pred_box)

    pred_detection = torch.stack(pred_detection).unsqueeze(0)
    pred_detection[:, :, 0, :] = pred_detection[:, :, 0, :] * 2
    pred_detection[:, :, 1, :] = pred_detection[:, :, 1, :] * 2

    ats_bounding_boxes = compute_ats_bounding_boxes(pred_detection[0], target[0]['boxes'])
    return ats_bounding_boxes

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_root',
    type=str,
    default='../data',
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
    default=1,
    help='batch size of dataloader')
parser.add_argument(
    '--optim_batch_size',
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
parser.add_argument(
    '--depth_net_path',
    type=str,
    default='./_depth_net.pth',
    help='path to depth net state dict')

args = parser.parse_args()
assert args.optimizer in ['sgd', 'adam']
assert args.model in [
    'basic', 'faster_rcnn', 'new_faster_rcnn',
    'detection_faster_rcnn']
assert args.dataset in [
    'basic', 'faster_rcnn', 'new_faster_rcnn']

def train(epoch):
    model.train()

    t = time.time()
    loader_len = len(train_dataloader)

    total_loss = 0.
    total_rpn_box_reg_loss = 0.
    total_rpn_cls_loss = 0.
    total_roi_box_reg_loss = 0.
    total_roi_cls_loss = 0.
    total_mask_loss = 0.
    total_mask_ts = 0.
    total_detection_ts = 0.
    batch_count = 0
    for batch_idx, data in enumerate(train_dataloader):
        original_target = copy.deepcopy(data[1])
        
        # Filter targets
        dis = torch.mean(data[1][0]['boxes'], dim=2) - torch.tensor([400., 400.])
        index_1 = torch.sqrt(torch.sum(torch.pow(dis, 2), dim=1)) < 300.
        index_2 = (data[1][0]['labels'] == 0) | (data[1][0]['labels'] == 2) |\
            (data[1][0]['labels'] == 4) | (data[1][0]['labels'] == 5)
        label_index = index_1 * index_2
        if torch.sum(label_index) == 0:
            continue

        imgs = torch.stack(data[0]).to(device)
        targets = data[1]
        losses = model(imgs, targets)

        loss = sum(l for l in losses.values())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        batch_count += len(imgs)
        if batch_count >= args.optim_batch_size or batch_idx == loader_len - 1:
            optimizer.step()
            optimizer.zero_grad()
            batch_count = 0

        # evaluate
        with torch.no_grad():
            model.eval()
            masks, detections = model(imgs, targets, return_result=True)
            mask_ts = get_mask_ts(masks.cpu(), original_target)
            detection_ts = get_detection_ts(detections, original_target)            

            total_loss += loss.cpu().item()
            total_rpn_box_reg_loss += losses['loss_rpn_box_reg'].item()
            total_rpn_cls_loss += losses['loss_objectness'].item()
            total_roi_box_reg_loss += losses['loss_box_reg'].item()
            total_roi_cls_loss += losses['loss_classifier'].item()
            total_mask_loss += losses['loss_mask'].item()
            total_mask_ts += mask_ts
            total_detection_ts += detection_ts
        model.train()

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

    total_results = {
        'loss': total_loss / loader_len,
        'rpn_box_reg_loss': total_rpn_box_reg_loss / loader_len,
        'rpn_cls_loss': total_rpn_cls_loss / loader_len,
        'roi_box_reg_loss': total_roi_box_reg_loss / loader_len,
        'roi_cls_loss': total_roi_cls_loss / loader_len,
        'mask_loss': total_mask_loss / loader_len,
        'mask_ts': total_mask_ts / loader_len,
        'detection_ts': total_detection_ts / loader_len
    }
    return total_results

def validate(epoch):
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
        total_mask_ts = 0.
        total_detection_ts = 0.
        for batch_idx, data in enumerate(val_dataloader):
            original_target = copy.deepcopy(data[1])
            
            # Filter targets
            dis = torch.mean(data[1][0]['boxes'], dim=2) - torch.tensor([400., 400.])
            index_1 = torch.sqrt(torch.sum(torch.pow(dis, 2), dim=1)) < 300.
            index_2 = (data[1][0]['labels'] == 0) | (data[1][0]['labels'] == 2) |\
                (data[1][0]['labels'] == 4) | (data[1][0]['labels'] == 5)
            label_index = index_1 * index_2
            if torch.sum(label_index) == 0:
                continue

            imgs = torch.stack(data[0]).to(device)
            targets = data[1]
            model.train()
            losses = model(imgs, targets)
            model.eval()
            masks, detections = model(imgs, targets, return_result=True)
            mask_ts = get_mask_ts(masks.cpu(), original_target)
            detection_ts = get_detection_ts(detections, original_target)

            loss = sum(l for l in losses.values())

            total_loss += loss.cpu().item()
            total_rpn_box_reg_loss += losses['loss_rpn_box_reg'].item()
            total_rpn_cls_loss += losses['loss_objectness'].item()
            total_roi_box_reg_loss += losses['loss_box_reg'].item()
            total_roi_cls_loss += losses['loss_classifier'].item()
            total_mask_loss += losses['loss_mask'].item()
            total_mask_ts += mask_ts
            total_detection_ts += detection_ts

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
            losses['loss_classifier'].item(), losses['loss_mask'].item(),
            mask_ts, detection_ts))
            t = time.time()

    total_results = {
        'loss': total_loss / loader_len,
        'rpn_box_reg_loss': total_rpn_box_reg_loss / loader_len,
        'rpn_cls_loss': total_rpn_cls_loss / loader_len,
        'roi_box_reg_loss': total_roi_box_reg_loss / loader_len,
        'roi_cls_loss': total_roi_cls_loss / loader_len,
        'mask_loss': total_mask_loss / loader_len,
        'mask_ts': total_mask_ts / loader_len,
        'detection_ts': total_detection_ts / loader_len
    }
    return total_results

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
#     device_count = torch.cuda.device_count()
#     if device_count > 1:
#         model = nn.DataParallel(model)
    model = model.to(device)
    # model.load_state_dict(torch.load('../test_v2/2020-05-09-05-41-04/epoch21.pth')['state_dict'])

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
        print('Starting Epoch {}\n'.format(epoch+1))

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
#         val_results = validate(epoch)
#         results['val_loss'].append(val_results['loss'])
#         results['val_rpn_box_reg_loss'].append(val_results['rpn_box_reg_loss'])
#         results['val_rpn_cls_loss'].append(val_results['rpn_cls_loss'])
#         results['val_roi_box_reg_loss'].append(val_results['roi_box_reg_loss'])
#         results['val_roi_cls_loss'].append(val_results['roi_cls_loss'])
#         results['val_mask_loss'].append(val_results['mask_loss'])
#         results['val_mask_ts'].append(val_results['mask_ts'])
#         results['val_detection_ts'].append(val_results['detection_ts'])
#         print(('\nTotal val loss: {:.3f} | '
#                'val rpn box regression loss: {:.3f} | '
#                'val rpn classifier loss: {:.3f} | '
#                'val roi box regression loss: {:.3f} | '
#                'val roi classifier loss: {:.3f} | '
#                'val mask loss: {:.3f} | '
#                'val mask ts: {:.3f} | '
#                'val detection ts: {:.3f}\n').format(
#             val_results['loss'], val_results['rpn_box_reg_loss'],
#             val_results['rpn_cls_loss'], val_results['roi_box_reg_loss'],
#             val_results['roi_cls_loss'], val_results['mask_loss'],
#             val_results['mask_ts'], val_results['detection_ts']))

        if not args.debug: # and last_val_loss > val_results['loss']:
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                # 'loss': val_results['loss']
            }
            print('\n***\nSaving epoch {}\n***\n'.format(epoch+1))
            torch.save(state, os.path.join(store_dir, f'epoch{epoch}.pth'))
            last_val_loss = val_results['loss']
            last_saved_epoch = epoch

    # Save results to a file
    if not args.debug:
        with open(store_file, 'wb') as f:
            pickle.dump(results, f)