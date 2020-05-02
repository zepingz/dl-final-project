from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


ipm_params = {
    
    'FRONT_LEFT': 
                    {
                        'quaternions' : [ 0.68316462   ,  -0.68338771  ,  0.17581486   ,  -0.18799863  ],
                        'translation' : [ 1.28400265   ,  0.31639086   ,  1.67877024   ], 
                        'intrinsics' :  [[879.03824732 ,  0.           ,  613.17597314 ],
                                         [  0.         ,  879.03824732 ,  524.14407205 ],
                                         [  0.         ,  0.           ,  1.           ] ],
                    },

    'FRONT':
                    {
                        'quaternions' : [ 0.50745829   , -0.49812866  ,  0.49496606   ,  -0.49934369 ],
                        'translation' : [ 1.50612211   , -0.03602647  ,  1.69421848   ],
                        'intrinsics' :  [[882.61644117 , 0.           ,  621.63358525 ],
                                         [  0.         , 882.61644117 ,  524.38397862 ],
                                         [  0.         , 0.           ,  1.           ] ],
                    },
    
    'FRONT_RIGHT':
                    {
                        'quaternions' : [ -0.19470424  ,  0.17808752   ,  -0.68312934  ,  0.68095909 ],
                        'translation' : [ 1.27320628   ,  -0.31664681  ,  1.68114556   ],
                        'intrinsics' :  [[880.41134027 ,  0.           ,  618.9494972  ],
                                         [  0.         ,  880.41134027 ,  521.38918482 ],
                                         [  0.         ,  0.           ,  1.           ] ],
                    },

    'BACK_LEFT':
                    {
                        'quaternions' : [ -0.67797289  ,  0.6871698    ,  0.19201452   ,  -0.1768143 ],
                        'translation' : [ 1.04338732   ,  0.31565584   ,  1.66400371   ],
                        'intrinsics' :  [[881.28264688 ,  0.           ,  612.29732111 ],
                                         [  0.         ,  881.28264688 ,  521.77447199 ],
                                         [  0.         ,  0.           ,  1.           ]  ],
                    },
    
    'BACK':
                    {
                        'quaternions' : [ -0.49033062   ,  0.50741961   ,  0.50819262   ,  -0.49379061 ],
                        'translation' : [ 0.81558292    ,  -0.00559198  ,  1.65395645   ],
                        'intrinsics' :  [[882.93018422  ,  0.           ,  616.45479905 ],
                                         [  0.          ,  882.93018422 ,  528.27123027 ],
                                         [  0.          ,  0.           ,   1.          ]  ],
                    },
    
    'BACK_RIGHT':
                    {
                        'quaternions' : [ -0.17126042  ,  0.1897148     ,  0.68851343   ,  -0.6786766 ],
                        'translation' : [ 1.04116266   ,  -0.31121292   ,  1.66718288   ],
                        'intrinsics' :  [[881.63835671 ,  0.            ,  607.66308183 ],
                                         [  0.         ,  881.63835671  ,  525.6185326  ],
                                         [  0.         ,  0.            ,  1.           ]  ],
                    },
    
}

class pdataset(Dataset):
    """Project dataset"""
    def __init__(self, list_file, data_root_path, img_size = [256, 306], bundle_size = 3):
        self.data_root_path = data_root_path
        self.img_size = img_size
        self.bundle_size = bundle_size
        with open(list_file) as file:
                self.frame_pathes = [x[:-1] for x in file.readlines()]
                
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((128, 416)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.frame_pathes)

    def __getitem__(self, item):
        # read camera intrinsics
        view = self.frame_pathes[item].split('CAM_')[1].split('.')[0]
        camparams = np.array(ipm_params[view]['intrinsics'])/4.0
        camparams[-1,-1] = 1.
        camparams = camparams.ravel()
        
        # read image bundle
        # slice the image into #bundle_size number of images
        frame_list = []
        #frame_list_ = []
        #print(self.frame_pathes[item])
        for i in range(self.bundle_size):
            sample_id = int(self.frame_pathes[item].split('/')[1].split('_')[1])
            old = self.frame_pathes[item].split('/')
            old[1] = old[1].split('_')[0]+'_'+str(sample_id+i)
            newpath = '/'.join(old)
            img_file = os.path.join(self.data_root_path, newpath)
            #print(img_file)
            # img_ = Image.open(img_file)
            # img_ = self.transform(img_)
            # frame_list_.append(img_)
            left = 0
            top = 0
            right = 306
            bottom = 200
            frame_list.append(np.array(Image.open(img_file).crop((left, top, right, bottom)) .resize((416, 128),resample = Image.BILINEAR)))
        frames = np.asarray(frame_list).astype(float).transpose(0, 3, 1, 2)
        # frames_ = (torch.stack(frame_list_, dim=0) * 255.).int().numpy().astype(float)
        
        return frames, camparams

if __name__ == "__main__":
    dataset = pdataset(list_file, data_root_path)
    for i, data in enumerate(dataset):
        if i == 20:
            print(data[1])
            break
