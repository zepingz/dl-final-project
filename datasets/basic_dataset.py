import os
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms

from utils.data import convert_map_to_road_map, convert_map_to_lane_map

img_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
]
corner_names = [
    'fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']

class BasicLabelledDataset(torch.utils.data.Dataset):
    scene_indices = range(106, 134)
    sample_indices = range(126)
    
    def __init__(self, root, img_transform=transforms.ToTensor(),
                 road_transform=transforms.ToTensor(), extra_info=True):
        self.root = root
        self.img_transform = img_transform
        self.road_transform = road_transform
        self.extra_info = extra_info
        self.anno = pd.read_csv(
            os.path.join(self.root, 'annotation.csv'))

    def __getitem__(self, index):
        scene_id = self.scene_indices[index // len(self.sample_indices)]
        sample_id = index % len(self.sample_indices)
        
        # Load camera images
        folder_path = os.path.join(
            self.root, f'scene_{scene_id}', f'sample_{sample_id}')
        img_paths = [
            os.path.join(folder_path, n) for n in img_names]
        imgs = [Image.open(p) for p in img_paths]
        if self.img_transform:
            imgs = torch.stack(
                [self.img_transform(img) for img in imgs])
            
        # Load bounding boxes
        data_entries = self.anno[(self.anno['scene']==scene_id)\
            & (self.anno['sample']==sample_id)]
        corners = data_entries[corner_names].to_numpy()
        categories = data_entries.category_id.to_numpy()
        
        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)
            
        # Load road images
        ego_path = os.path.join(folder_path, 'ego.png')
        ego_img = Image.open(ego_path)
        ego_img = transforms.functional.to_tensor(ego_img)
        target_ego_img = self.road_transform(ego_img)
        road_img = convert_map_to_road_map(ego_img).unsqueeze(0)
        target_road_img = convert_map_to_road_map(target_ego_img).unsqueeze(0).float()
        
        # Load extra info
        if self.extra_info:
            extra = {}
            extra['action'] = torch.as_tensor(
                data_entries.action_id.to_numpy())
            extra['ego_image'] = ego_img
            # You can change the binary_lane to False to get a lane with
            extra['lane_image'] = convert_map_to_lane_map(
                ego_img, binary_lane=True)
            return imgs, target, road_img, target_road_img, extra
        else:
            return imgs, target, road_img, target_road_img
    
    def __len__(self):
        return len(self.scene_indices) * len(self.sample_indices)