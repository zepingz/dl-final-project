import os
from glob import glob
from PIL import Image

import torch
import torchvision.transforms as transforms

from datasets.basic_dataset import img_names

root = '../shared/dl/data'
folder_paths = [path for path in\
    glob(os.path.join(root, '*')) if 'csv' not in path]

total_num = 0
mean = torch.zeros(3)
std = torch.zeros(3)
for folder_path in folder_paths:
    print(folder_path)
    sample_paths = glob(os.path.join(folder_path, '*'))
    for sample_path in sample_paths:
        for img_name in img_names:
            img = Image.open(os.path.join(sample_path, img_name))
            img = transforms.ToTensor()(img).view(3, -1)
            mean += img.mean(1)
            std += img.std(1)
        total_num += len(img_names)
        
mean /= total_num
std /= total_num
print('mean:', mean) # [0.5459, 0.5968, 0.6303]
print('std:', std) # [0.3178, 0.3246, 0.3278]