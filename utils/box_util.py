# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Last modified: Jul 2019
"""
from __future__ import print_function

import numpy as np
from scipy.spatial import ConvexHull

import pyviz3d.visualizer as viz
import os

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    # rect1_ = [(corners1[i, 0], corners1[i, 1]) for i in range(3, -1, -1)]  # 3 2 1 0
    # rect2_ = [(corners2[i, 0], corners2[i, 1]) for i in range(3, -1, -1)]
    if corners1.shape[0] == 8:
        rect1 = [(corners1[i, 0], corners1[i, 1]) for i in [0, 1, 2, 3]]  # 3 2 1 0
    else:
        rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(corners1.shape[0]//2, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 1]) for i in [0, 1, 2, 3]]

    # rect1 = [((corners1[i, 0] + corners1[i + 4, 0]) / 2.0,
    #           (corners1[i, 1] + corners1[i + 4, 1]) / 2.0) for i in [0, 1, 2, 3]]  # 3 2 1 0
    # rect2 = [((corners2[i, 0] + corners2[i + 4, 0]) / 2.0,
    #           (corners2[i, 1] + corners2[i + 4, 1]) / 2.0) for i in [0, 1, 2, 3]]

    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)

    zmax = min(corners1[-1, 2], corners2[-1, 2])
    zmin = max(corners1[0, 2], corners2[0, 2])
    inter_vol = inter_area * max(0.0, zmax - zmin)

    vol1 = area1 * (corners1[-1, 2] - corners1[0, 2]) # box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def box2d_iou(box1, box2):
    ''' Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    '''
    return get_iou({'x1': box1[0], 'y1': box1[1], 'x2': box1[2], 'y2': box1[3]}, \
                   {'x1': box2[0], 'y1': box2[1], 'x2': box2[2], 'y2': box2[3]})


# -----------------------------------------------------------
# Convert from box parameters to
# -----------------------------------------------------------
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_3d_box_batch(box_size, heading_angle, center):
    ''' box_size: [x1,x2,...,xn,3]
        heading_angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    '''
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = np.concatenate((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = np.concatenate((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d


if __name__ == '__main__':

    # Function for polygon ploting
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt


    def plot_polys(plist, scale=500.0):
        fig, ax = plt.subplots()
        patches = []
        for p in plist:
            poly = Polygon(np.array(p) / scale, True)
            patches.append(poly)


    pc = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.5)
    colors = 100 * np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    plt.show()

    # Demo on ConvexHull
    points = np.random.rand(30, 2)  # 30 random points in 2-D
    hull = ConvexHull(points)
    # **In 2D "volume" is is area, "area" is perimeter
    print(('Hull area: ', hull.volume))
    for simplex in hull.simplices:
        print(simplex)

    # Demo on convex hull overlaps
    sub_poly = [(0, 0), (300, 0), (300, 300), (0, 300)]
    clip_poly = [(150, 150), (300, 300), (150, 450), (0, 300)]
    inter_poly = polygon_clip(sub_poly, clip_poly)
    print(poly_area(np.array(inter_poly)[:, 0], np.array(inter_poly)[:, 1]))

    # Test convex hull interaction function
    rect1 = [(50, 0), (50, 300), (300, 300), (300, 0)]
    rect2 = [(150, 150), (300, 300), (150, 450), (0, 300)]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
    if inter is not None:
        print(poly_area(np.array(inter)[:, 0], np.array(inter)[:, 1]))

    print('------------------')
    rect1 = [(0.30026005199835404, 8.9408694211408424), \
             (-1.1571105364358421, 9.4686676477075533), \
             (0.1777082043006144, 13.154404877812102), \
             (1.6350787927348105, 12.626606651245391)]
    rect1 = [rect1[0], rect1[3], rect1[2], rect1[1]]
    rect2 = [(0.23908745901608636, 8.8551095691132886), \
             (-1.2771419487733995, 9.4269062966181956), \
             (0.13138836963152717, 13.161896351296868), \
             (1.647617777421013, 12.590099623791961)]
    rect2 = [rect2[0], rect2[3], rect2[2], rect2[1]]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))


def get_rotated_bounds(bb_bounds, rotation):
    """computes the rotated corner points of a bounding box specified by bounds, centered at origin i.e. trans=0
       And then returns the axis aligned bounds of that rotated bounding box.
    """
    corner_000 = rotation @ np.array([-bb_bounds[0], -bb_bounds[1], -bb_bounds[2]])
    corner_001 = rotation @ np.array([-bb_bounds[0], -bb_bounds[1], bb_bounds[2]])
    corner_010 = rotation @ np.array([-bb_bounds[0], bb_bounds[1], -bb_bounds[2]])
    corner_011 = rotation @ np.array([-bb_bounds[0], bb_bounds[1], bb_bounds[2]])
    corner_100 = rotation @ np.array([bb_bounds[0], -bb_bounds[1], -bb_bounds[2]])
    corner_101 = rotation @ np.array([bb_bounds[0], -bb_bounds[1], bb_bounds[2]])
    corner_110 = rotation @ np.array([bb_bounds[0], bb_bounds[1], -bb_bounds[2]])
    corner_111 = rotation @ np.array([bb_bounds[0], bb_bounds[1], bb_bounds[2]])
    corners = [corner_000, corner_001, corner_010, corner_011, corner_100, corner_101, corner_110, corner_111]
    bounds = np.array([0.0, 0.0, 0.0])
    for corner in corners:
        for j in range(3):
            if corner[j] > bounds[j]:
                bounds[j] = corner[j]
    return bounds


def get_oriented_corners(bb_bounds, rotation, translation):
    """computes the rotated corner points of a bounding box specified by bounds
       returns:        corners1: numpy array (8,3), assume up direction is negative Y

      10---11
     /    /
    00---01

    """

    corner_000 = np.array([-bb_bounds[0], -bb_bounds[1], -bb_bounds[2]])
    corner_100 = np.array([bb_bounds[0], -bb_bounds[1], -bb_bounds[2]])
    corner_110 = np.array([bb_bounds[0], bb_bounds[1], -bb_bounds[2]])
    corner_010 = np.array([-bb_bounds[0], bb_bounds[1], -bb_bounds[2]])

    corner_001 = np.array([-bb_bounds[0], -bb_bounds[1], bb_bounds[2]])
    corner_101 = np.array([bb_bounds[0], -bb_bounds[1], bb_bounds[2]])
    corner_111 = np.array([bb_bounds[0], bb_bounds[1], bb_bounds[2]])
    corner_011 = np.array([-bb_bounds[0], bb_bounds[1], bb_bounds[2]])

    corners = [corner_000, corner_100, corner_110, corner_010,
               corner_001, corner_101, corner_111, corner_011]
    corners = [rotation @ c + translation for c in corners]
    return np.concatenate([corners], axis=0)

if __name__ == "__main__":
    # debug
    corners1_ = np.array([[ 0.40853998,  0.2875091 ,  0.13545305],
       [ 0.42311999, -1.91749094,  0.13545305],
       [ 0.43811986, -1.92431091,  0.13545305],
       [ 2.14811989, -1.86443089,  0.13545305],
       [ 2.16312   , -1.857491  ,  0.13545305],
       [ 2.20811984, -1.82749103,  0.13545305],
       [ 2.23812005, -1.69210099,  0.13545305],
       [ 2.24270007, -1.66249083,  0.13545305],
       [ 2.23812005, -1.29024099,  0.13545305],
       [ 2.19311997, -0.29222094,  0.13545305],
       [ 2.16312   ,  0.2875091 ,  0.13545305],
       [ 2.01680991,  0.3475091 ,  0.13545305],
       [ 1.99812004,  0.35462005,  0.13545305],
       [ 0.49027988,  0.37750907,  0.13545305],
       [ 0.44906994,  0.3475091 ,  0.13545305],
       [ 0.40853998,  0.2875091 ,  0.72658222],
       [ 0.42311999, -1.91749094,  0.72658222],
       [ 0.43811986, -1.92431091,  0.72658222],
       [ 2.14811989, -1.86443089,  0.72658222],
       [ 2.16312   , -1.857491  ,  0.72658222],
       [ 2.20811984, -1.82749103,  0.72658222],
       [ 2.23812005, -1.69210099,  0.72658222],
       [ 2.24270007, -1.66249083,  0.72658222],
       [ 2.23812005, -1.29024099,  0.72658222],
       [ 2.19311997, -0.29222094,  0.72658222],
       [ 2.16312   ,  0.2875091 ,  0.72658222],
       [ 2.01680991,  0.3475091 ,  0.72658222],
       [ 1.99812004,  0.35462005,  0.72658222],
       [ 0.49027988,  0.37750907,  0.72658222],
       [ 0.44906994,  0.3475091 ,  0.72658222]])

    corners2_ = np.array([[ 2.21297868, -1.82235349,  0.19845089],
       [ 2.12342826,  0.33659047,  0.19845114],
       [ 0.3918886 ,  0.26476827,  0.19849851],
       [ 0.48143901, -1.89417568,  0.19849826],
       [ 2.21299212, -1.82235299,  0.69073097],
       [ 2.12344171,  0.33659097,  0.69073122],
       [ 0.39190205,  0.26476878,  0.69077859],
       [ 0.48145246, -1.89417518,  0.69077835]])

    rect1_ = np.array([(0.4085399779010701, 0.2875090985587416,0),
 (0.4490699442554402, 0.3475091009429274,0),
 (0.490279880397408, 0.3775090723326979,0),
 (1.9981200369525838, 0.35462005427392107,0),
 (2.0168099077869344, 0.3475091009429274,0),
 (2.163119998805611, 0.2875090985587416,0),
 (2.1931199701953816, -0.2922209353157701,0),
 (2.238120046489327, -1.2902409882256212,0),
 (2.242700067393868, -1.6624908299156846,0),
 (2.238120046489327, -1.692100986928624,0),
 (2.2081198366809773, -1.827491030187291,0),
 (2.163119998805611, -1.8574910015770616,0),
 (2.1481198939014363, -1.8644308895775499,0),
 (0.43811985575446366, -1.9243109078118028,0),
 (0.42311998926886796, -1.9174909443566026,0),
 (0.4085399779010701, 0.2875090985587416,0)]
)

    rect2_ = np.array([(2.212978676641883, -1.8223534877194096,0),
 (2.1234282648817815, 0.33659046788072966,0),
 (0.39188859861031067, 0.26476827457975527,0),
 (0.48143901037041226, -1.894175681020384,0)])

    inter, inter_area = convex_hull_intersection(rect1_, rect2_)

    v = viz.Visualizer()
    v.add_polyline(f'box1', positions=corners1_, edge_width=0.01, color=np.array([255, 0, 0]))
    v.add_polyline(f'box2', positions=corners2_, edge_width=0.01, color=np.array([0, 255, 0]))
    v.add_polyline(f'rect1', positions=rect1_, edge_width=0.01, color=np.array([100, 0, 0]))
    v.add_polyline(f'rect2', positions=rect2_, edge_width=0.01, color=np.array([0, 100, 0]))
    v.save(os.path.join('viz_debug', '3d_iou'))

    box3d_iou(corners1, corners2)
