import os
import cv2
import math
import kornia
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch

def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    ax.plot(point_squence.T[0], point_squence.T[1], color=color)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return torch.tensor([qx, qy])


class BEV():
    def __init__( self, data_root, scene_id, sample_id, ipm_params ):
        self.data_root = data_root
        self.scene_id = scene_id
        self.sample_id = sample_id
        self.ipm_params = ipm_params
        self.bevs = self._getBEV()
        self.combined_bev = self._combine()

    def _getBEV( self ):
        bevs = {}
        for view in self.ipm_params:
            path = os.path.join(
                self.data_root,
                'scene_%d' % self.scene_id,
                'sample_%d' % self.sample_id,
                'CAM_%s.jpeg' % view)
            points_src, points_dst = self.ipm_params[view]
            bevs[view] = _BEV(path, points_src, points_dst)
        return bevs

    def _combine( self ):
        out = None
        priority_queue = ['FRONT','BACK','FRONT_LEFT','FRONT_RIGHT','BACK_LEFT','BACK_RIGHT']
        for view in priority_queue:
            if view in self.bevs:
                if out is None:
                    out = self.bevs[view].dst_img.copy()
                else:
                    new_layer = self.bevs[view].dst_img
                    mask = (out.sum(2) == 0).astype(np.uint8)
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    out += new_layer * mask
        return out

    def visualize( self, view = None, anno_path = 'data/annotation.csv' ):

        corner_names = ['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']
        colors = ['c', 'r', 'b', 'g']

        # Get target
        anno = pd.read_csv( anno_path )
        data_entries = anno[(anno['scene']==self.scene_id)\
            & (anno['sample']==self.sample_id)]
        corners = data_entries[corner_names].to_numpy()
        targets = data_entries['category_id'].values.tolist()

        corners = torch.tensor(
            data_entries[corner_names].to_numpy(), dtype=torch.float).view(-1, 2, 4)
        corners[:, 0, :] = corners[:, 0, :] * 10 + 400
        corners[:, 1, :] = -corners[:, 1, :] * 10 + 400

        # Rotate
        for i in range(len(corners)):
            for j in range(4):
                corners[i, :, j] = rotate((400, 400), corners[i, :, j], -math.pi / 2)

        if view is not None:
            src_img = self.bevs[view].src_img
            dst_img = self.bevs[view].dst_img

            # Visualization
            fig, axs = plt.subplots(1, 2, figsize=(20, 15))
            axs = axs.ravel()

            axs[0].set_title('image source')
            axs[0].imshow(src_img)

            axs[1].set_title('image destination')
            axs[1].imshow(dst_img)
            axs[1].plot(400, 400, 'x', color='red')
            color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
            #print(targets)
            for i, bb in enumerate(corners):
                draw_box(axs[1], bb, color=color_list[targets[i]])
                x,y = bb.mean(dim = 1).numpy()
                plt.text(x, y, '(%s,%s)'%(str(round(x,2)),str(round(y,2))), color=color_list[targets[i]])

            points_src, points_dst = ipm_params[view]
            for i in range(points_src.size(1)):
                axs[0].scatter(points_src[0, i, 0], points_src[0, i, 1], color=colors[i])
                axs[1].scatter(points_dst[0, i, 0], points_dst[0, i, 1], color=colors[i])

        else:
            fig, axs = plt.subplots(figsize=(12, 12))
            axs.imshow(self.combined_bev)
            #print(targets)
            axs.plot(400, 400, 'x', color='red')
            color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
            for i, bb in enumerate(corners):
                draw_box(axs, bb, color=color_list[targets[i]])
                x,y = bb.mean(dim = 1).numpy()
                plt.text(x, y, '(%s,%s)'%(str(round(x,2)),str(round(y,2))), color=color_list[targets[i]])


    def save ( self, path, view = None ):
        if view is not None:
            plt.imsave( path, self.bevs[view].dst_img)
        else:
            plt.imsave( path, self.combined_bev )


class _BEV():
    def __init__( self, path, points_src, points_dst ):
        self.path =  path
        self.points_src = points_src
        self.points_dst = points_dst
        self.src_img =  self._read()
        self.dst_img = self._convert()

    def _read( self ):
        return cv2.cvtColor( cv2.imread( self.path ) , cv2.COLOR_BGR2RGB )

    def _convert( self ):

        src_img = kornia.image_to_tensor( self.src_img, keepdim = False )

        dst_h, dst_w = 800, 800

        # Compute perspective transform
        M = kornia.get_perspective_transform(self.points_src, self.points_dst)

        # Image to BEV transformation
        dst_img = kornia.warp_perspective(
            src_img.float(), M, dsize=(dst_h, dst_w), flags='bilinear', border_mode='zeros')

        # remove unwanted portion of BEV image. e.g for FRONT view dst point should not be higher than 450.
        if 'FRONT' in self.path:
            dst_img[:, :, 400:, :] = 0

        if 'BACK' in self.path:
            dst_img[:, :, :400, :] = 0

        if 'LEFT' in self.path:
            dst_img[:, :, :, 400:] = 0

        if 'RIGHT' in self.path:
            dst_img[:, :, :, :400] = 0

        dst_img = kornia.tensor_to_image(dst_img.byte())
        return dst_img
