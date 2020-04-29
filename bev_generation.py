import os
import argparse

import torch

from utils.bev import BEV

ipm_params = {
    # calibrated using SCENE 111 SAMPLE 11 image
    'FRONT': (torch.tensor([[ [100.,180.], [123.,163.], [267., 170.], [207., 143.],]]),
              torch.tensor([[[389, 323], [389, 284], [442, 305.], [461,74.],]])),
    # calibrated using SCENE 124 SAMPLE 0 FRONT_RIGHT image
    'FRONT_RIGHT': (torch.tensor([[[82., 191.], [125., 174.], [257,219.], [264.,190.],]]),
                    torch.tensor([[[438., 342.], [455., 342.], [440, 389], [455, 390],]]) ),
    # calibrated using SCENE 124 SAMPLE 0 FRONT_RIGHT image
    'BACK_RIGHT': (torch.tensor([[[31., 191.], [34., 221.], [188,169.], [229.,183.],]]),
                   torch.tensor([[[456., 405.], [440., 405.], [460, 460], [441, 460],]]) ),
    # calibrated using SCENE 126 SAMPLE 70 FRONT_LEFT image
    'FRONT_LEFT': (torch.tensor([[[228., 191.], [52., 219.], [202, 174.], [44.,192.],]]),
                   torch.tensor([[[368., 345.], [367., 420.], [348, 345], [347, 420],]]) ),
    # calibrated using SCENE 110 SAMPLE 6 BACK_LEFT image
    'BACK_LEFT': (torch.tensor([[[273., 208.], [274., 190.], [48.,170], [94.,166.],]]),
                  torch.tensor([[[358., 400.], [345., 400.], [358, 511],[345, 511], ]]) ),
    # calibrated using SCENE 111 SAMPLE 28 image
    'BACK': (torch.tensor([[[32., 153.], [192., 167.], [179.,153.], [93,135],]]),
             torch.tensor([[[464.,540], [391., 484.], [391.,525],[490,769]]]) ),
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../data', help='data root')
args = parser.parse_args()

if __name__ == '__main__':
    for scene_id in range(0, 134):
        for sample_id in range(0, 125):
            print('scene_{} sample_{}'.format(scene_id, sample_id))
            bev = BEV(args.data_root, scene_id, sample_id, ipm_params)
            path = os.path.join(
                args.data_root,
                'scene_{}'.format(scene_id),
                'sample_{}'.format(sample_id), 'BEV.jpeg')
            bev.save(path)
