import os
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms

from utils.target_transforms import TargetResize
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

class FasterRCNNLabelledDataset(torch.utils.data.Dataset):
    scene_indices = range(106, 134)
    sample_indices = range(126)
    
    def __init__(self, root, img_transform=transforms.ToTensor(),
                 target_transform=TargetResize(800, (512, 920)),
                 extra_info=True):
        self.root = root
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.extra_info = extra_info
        self.anno = pd.read_csv(
            os.path.join(self.root, 'annotation.csv'))
        
        self.concat_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad((1, 0)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        scene_id = self.scene_indices[index // len(self.sample_indices)]
        sample_id = index % len(self.sample_indices)
        
        # Load camera images
        folder_path = os.path.join(
            self.root, f'scene_{scene_id}', f'sample_{sample_id}')
        img_paths = [
            os.path.join(folder_path, n) for n in img_names]
        img = [Image.open(p) for p in img_paths]
        if self.img_transform:
            img = torch.stack(
                [self.img_transform(i) for i in img])
            
        # Concatenat images into a big image
        img = torch.cat(
            (torch.cat((img[0], img[1], img[2]), dim=2),
             torch.cat((img[3], img[4], img[5]), dim=2)), dim=1)
        
        # Pad the concatenated image
        img = self.concat_img_transform(img)
            
        # Load bounding boxes
        data_entries = self.anno[(self.anno['scene']==scene_id)\
            & (self.anno['sample']==sample_id)]
        corners = data_entries[corner_names].to_numpy()
        corners = torch.tensor(
            data_entries[corner_names].to_numpy(), dtype=torch.float).view(-1, 2, 4)
        corners[:, 0, :] = corners[:, 0, :] * 10 + 400
        corners[:, 1, :] = -corners[:, 1, :] * 10 + 400
        min_coordinates, _ = torch.min(corners, 2)
        max_coordinates, _ = torch.max(corners, 2)
        corners = torch.cat([min_coordinates, max_coordinates], 1)
        categories = data_entries.category_id.to_numpy()
            
        # Load road images
        ego_path = os.path.join(folder_path, 'ego.png')
        ego_img = Image.open(ego_path)
        ego_img = transforms.functional.to_tensor(ego_img)
        road_img = convert_map_to_road_map(ego_img).unsqueeze(0).float()
        
        # Make target
        original_target = {}
        original_target['boxes'] = corners
        original_target['labels'] = torch.as_tensor(categories)
        original_target["masks"] = road_img # target_road_img
        
        if self.target_transform:
            target = self.target_transform(original_target)
        else:
            target = original_target
        
        # Load extra info
        if self.extra_info:
            extra = {}
            extra['action'] = torch.as_tensor(
                data_entries.action_id.to_numpy())
            extra['ego_image'] = ego_img
            # You can change the binary_lane to False to get a lane with
            extra['lane_image'] = convert_map_to_lane_map(
                ego_img, binary_lane=True)
            return img, target, original_target, extra
        else:
            return img, target, original_target
    
    def __len__(self):
        return len(self.scene_indices) * len(self.sample_indices)