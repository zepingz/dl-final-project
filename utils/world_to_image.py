import math
import numpy as np

import torch

intrinsics = {
    'FRONT_LEFT': np.array([[879.03824732 / 4., 0.0, 613.17597314 / 4., 0.0],
                            [0.0, 879.03824732 / 4., 524.14407205 / 4., 0.0],
                            [0.0, 0.0, 1.0, 0.0]]),
    'FRONT': np.array([[882.61644117 / 4., 0.0, 621.63358525 / 4., 0.0],
                       [0.0, 882.61644117 / 4., 524.38397862 / 4., 0.0],
                       [0.0, 0.0, 1.0, 0.0]]),
    'FRONT_RIGHT': np.array([[880.41134027 / 4., 0.0, 618.9494972 / 4., 0.0],
                             [0.0, 880.41134027 / 4., 521.38918482 / 4., 0.0],
                             [0.0, 0.0, 1.0, 0.0]]),
    'BACK_LEFT': np.array([[881.28264688 / 4., 0.0, 612.29732111 / 4., 0.0],
                           [0.0, 881.28264688 / 4., 521.77447199 / 4., 0.0],
                           [0.0, 0.0, 1.0, 0.0]]),
    'BACK': np.array([[882.93018422 / 4., 0.0, 616.45479905 / 4., 0.0],
                      [0.0, 882.93018422 / 4., 528.27123027 / 4., 0.0],
                      [0.0, 0.0, 1.0, 0.0]]),
    'BACK_RIGHT': np.array([[881.63835671 / 4., 0.0, 607.66308183 / 4., 0.0],
                            [0.0, 881.63835671 / 4., 525.6185326 / 4., 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
}

extrinsics = {
    'FRONT_LEFT': np.array([[0.86748958, 0.01656881, 0.49717935, 1.28400265 / 4.],
                            [-0.49717391, -0.00473755, 0.86763798, 0.31639086 / 4.],
                            [0.01673114, -0.9998515, 0.00412781, 1.67877024 / 4.],
                            [0., 0., 0., 1.]]),
    'FRONT': np.array([[0.01131076, 0.01367888, 0.99984246, 1.50612211 / 4.],
                       [-0.99992415, 0.00502913, 0.01124289, -0.03602647 / 4.],
                       [-0.00487454, -0.99989379, 0.01373473, 1.69421848 / 4.],
                       [0., 0., 0., 1.]]),
    'FRONT_RIGHT': np.array([[-0.86074962, 0.02185771, 0.50855908, 1.27320628 / 4.],
                             [-0.50848696, 0.00915503, -0.86102102, -0.31664681 / 4.],
                             [-0.02347582, -0.99971917, 0.00323418, 1.68114556 / 4.],
                             [0., 0., 0., 1.]]),
    'BACK_LEFT': np.array([[0.86373186, 0.02414298, -0.50337301, 1.04338732 / 4.],
                           [0.5036526, -0.00694894, 0.86387833, 0.31565584 / 4.],
                           [ 0.01735869, -0.99968436, -0.01816169, 1.66400371 / 4.],
                           [0., 0., 0., 1.]]),
    'BACK': np.array([[-0.00419018, 0.03149288, -0.99949519, 0.81558292 / 4.],
                      [ 0.99998743, -0.00262, -0.0042748, -0.00559198 / 4.],
                      [-0.00275331, -0.99950054, -0.0314815, 1.65395645 / 4.],
                      [0., 0., 0., 1.]]),
    'BACK_RIGHT': np.array([[-0.869353, 0.02878223, -0.49335277, 1.04116266 / 4.],
                            [0.49371584, 0.00678742, -0.8695968, -0.31121292 / 4.],
                            [-0.02168035, -0.99956266, -0.0201109, 1.66718288 / 4.],
                            [0., 0., 0., 1.]])
}

angles = {
    'FRONT_LEFT': math.radians(30),
    'FRONT': math.radians(90),
    'FRONT_RIGHT': math.radians(150),
    'BACK_LEFT': math.radians(-30),
    'BACK': math.radians(-90),
    'BACK_RIGHT': math.radians(-150)
}

img_index = {
    'FRONT_LEFT': 0,
    'FRONT': 1,
    'FRONT_RIGHT': 2,
    'BACK_LEFT': 3,
    'BACK': 4,
    'BACK_RIGHT': 5}

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

def world_to_image(points, view, image_center=200., bottom=-0.5, top=1.):
    points = (points - image_center) / 10.
    points = points.permute(0, 2, 1)
    xyz_mat = torch.cat((
        points.repeat(1, 2, 1),
        torch.tensor([[bottom, 1.], [top, 1.]]).repeat(1, 4).view(1, -1, 2).repeat(len(points), 1, 1)), dim=2)
    uvw_mat = intrinsics[view].dot(extrinsics[view]).dot(xyz_mat.transpose(2, 1))
    us = uvw_mat[0] / uvw_mat[2]
    vs = uvw_mat[1] / uvw_mat[2]
    
    # Rotate
    ox, oy = 306 / 2, 256 / 2
    qx = ox + math.cos(angles[view]) * (us - ox) - math.sin(angles[view]) * (vs - oy)
    qy = oy + math.sin(angles[view]) * (us - ox) + math.cos(angles[view]) * (vs - oy)
    
    min_x, max_x = np.min(qx, axis=1), np.max(qx, axis=1)
    min_y, max_y = np.min(qy, axis=1), np.max(qy, axis=1)
    return min_x, max_x, min_y, max_y

def get_view_point(corners, image_center=200.):
    points = {
        'FRONT_LEFT': [],
        'FRONT': [],
        'FRONT_RIGHT': [],
        'BACK_LEFT': [],
        'BACK': [],
        'BACK_RIGHT': []}
    points_index = {
        'FRONT_LEFT': [],
        'FRONT': [],
        'FRONT_RIGHT': [],
        'BACK_LEFT': [],
        'BACK': [],
        'BACK_RIGHT': []}
    for i, corner in enumerate(corners):
    #     min_coordinates, _ = torch.min(corner, 1)
    #     max_coordinates, _ = torch.max(corner, 1)
    #     corner = torch.cat([min_coordinates, max_coordinates], 0)
        corner = torch.stack([
            torch.stack([corner[0], corner[0], corner[2], corner[2]]),
            torch.stack([corner[1], corner[1], corner[3], corner[3]])])
        center = torch.mean(corner, dim=1)
        degree = math.degrees(math.atan2(center[0].item()-image_center, center[1].item()-image_center))
        # FRONT
        if degree > 60. and degree < 120.:
            # Rotate
            for j in range(4):
                corner[:, j] = rotate((image_center, image_center), corner[:, j], -angles['FRONT'])
            points['FRONT'].append(corner)
            points_index['FRONT'].append(i)
        # FRONT LEFT
        elif degree > 120. and degree < 180.:
            # Rotate
            for j in range(4):
                corner[:, j] = rotate((image_center, image_center), corner[:, j], -angles['FRONT_LEFT'])
            points['FRONT_LEFT'].append(corner)
            points_index['FRONT_LEFT'].append(i)
        # FRONT RIGHT
        elif degree > 0. and degree < 60.:
            # Rotate
            for j in range(4):
                corner[:, j] = rotate((image_center, image_center), corner[:, j], -angles['FRONT_RIGHT'])
            points['FRONT_RIGHT'].append(corner)
            points_index['FRONT_RIGHT'].append(i)
        # BACK
        elif degree > -120. and degree < -60.:
            # Rotate
            for j in range(4):
                corner[:, j] = rotate((image_center, image_center), corner[:, j], -angles['BACK'])
            points['BACK'].append(corner)
            points_index['BACK'].append(i)
        # BACK LEFT
        elif degree > -180. and degree < -120.:
            # Rotate
            for j in range(4):
                corner[:, j] = rotate((image_center, image_center), corner[:, j], -angles['BACK_LEFT'])
            points['BACK_LEFT'].append(corner)
            points_index['BACK_LEFT'].append(i)
        # BACK RIGHT
        elif degree > -60. and degree < 0.:
            # Rotate
            for j in range(4):
                corner[:, j] = rotate((image_center, image_center), corner[:, j], -angles['BACK_RIGHT'])
            points['BACK_RIGHT'].append(corner)
            points_index['BACK_RIGHT'].append(i)
            
    return points, points_index
