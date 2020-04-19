import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])

    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)
    
def new_draw_box(ax, corners, color):
    rect = Rectangle(
        corners[:2], corners[2]-corners[0], corners[3]-corners[1],
        color=color, fill=False)
    ax.add_patch(rect)
    
# imgs, target, original_target, extra = next(iter(train_dataloader))
# visualize_target(target[0]['masks'][0], target[0])
# visualize_target(original_target[0]['masks'][0], original_target[0])
def visualize_target(img, target):
    fig, ax = plt.subplots()
    color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
    ax.imshow(img, cmap='binary')
    ax.plot(img.shape[1] // 2, img.shape[0] // 2, 'x', color='red')
    for i, bb in enumerate(target['boxes']):
        new_draw_box(ax, bb, color=color_list[target['labels'][i]])