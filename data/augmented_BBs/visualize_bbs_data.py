import sys
sys.path.append('.')

import open3d as o3d
from dataprocessing.scannet import scannet_color_map, SEMANTIC_VALID_CLASS_IDS, SEMANTIC_VALID_CLASS_IDS_torch
import config_loader as cfg_loader
import pyviz3d.visualizer as viz
from dataprocessing import scannet
import os
import numpy as np
from utils.util import get_bbs_lines
import configargparse

# Argument lists
parser = configargparse.ArgumentParser()
parser.add_argument ("--data", type=str, default='noisy1',
                    help='Data can be one of noisy1, dropout10, etc')
parser.add_argument ("--scene_name", type=str, default='scene0293_00',
                    help='Scene to be processed')
parser.add_argument ("--data_path", type=str, default='/BS/atran2/work/tmp/for_webpage/scannet_boxes_data/'
                    help='Path to the augmentated boxes data')
cfg = parser.parse_args(args)


# Get the rgb point cloud, original labels of the scene
scene, labels = scannet.process_scene(scene_name, 'train', cfg, do_augmentations=False)

# Specify a data set (eg. noisy1, dropout10)
data_name=cfg.data
scene_name = cfg.scene_name

# Load the instance BB of the scene
boxes_data_path = os.path.join ("", data_name)
box_info_pth = os.path.join (cfg.data_path, scene_name + '.npy')
boxes = np.load (box_info_pth, allow_pickle=True).item ()

v = viz.Visualizer()

v.add_points ("Scene RGB", scene["positions"], scene['colors'] * 255, point_size=25, visible=False)

min_corners = boxes["min_corner"] # List of min corners of instances
max_corners = boxes["max_corner"] # List of max corners of instances
semantic_ids = boxes ["semantic_id"] # the list containing semantic id of each box

# Visualize each instance
for instance_id in range(len(semantic_ids)):
    min_corner = min_corners[instance_id][None] # shape 1x3
    max_corner = max_corners[instance_id][None] # shape 1x3
    semantic_id = semantic_ids[instance_id] 
    
    # Get the 12 edges of the box
    bb_centers = (max_corner + min_corner) / 2
    bb_bounds = max_corner - bb_centers
    start, end = get_bbs_lines(bb_centers, bb_bounds)
    semantic_color = scannet.scannet_color_map [semantic_id]
    semantic_name = scannet.scannet_class_names [semantic_id]
    lines_color = np.stack ([semantic_color for _ in range (12)])
    
    # Draw the box using piviz
    v.add_lines(semantic_name+';instance_id='+str(instance_id), start, end, lines_color, visible=False)

visualize_path = os.path.join ("data/augmented_BBs/visualize/", data_name)
os.makedirs(visualize_path, exist_ok=True)
v.save(os.path.join(visualize_path, scene_name))