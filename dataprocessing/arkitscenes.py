import sys
sys.path.append('.')

import os
import random
import json
import csv
from types import MemberDescriptorType

import numpy as np
import open3d as o3d
import pyviz3d.visualizer as viz
import config_loader as cfg_loader
import dataprocessing.augmentation as augmentation
from scipy.stats import rankdata
import torch
import math
import pyviz3d.visualizer as viz
import quaternion as q
from scipy.spatial.transform import Rotation as R

# for visulaization with pyviz3d
arkitscenes_color_map = np.array([
  (0, 0, 0),  # unlabeled
  (174, 199, 232),  # wall          1
  (152, 223, 138),  # floor         2
  (31, 119, 180),  # cabinet        3
  (255, 187, 120),  # bed           4
  (188, 189, 34),  # chair          5
  (140, 86, 75),  # sofa            6
  (255, 152, 150),  # table         7
  (214, 39, 40),  # door            8
  (197, 176, 213),  # window        9
  (148, 103, 189),  # bookshelf     10
  (196, 156, 148),  # picture       11
  (23, 190, 207),  # counter        12
  (178, 76, 76),
  (247, 182, 210),  # desk          14
  (66, 188, 102),
  (219, 219, 141),  # curtain       15
  (140, 57, 197),    # 16
  (202, 185, 52),   # 17
  (51, 176, 203),   # 18
  (200, 54, 131),   # 19
  (92, 193, 61),    # 20
  (78, 71, 183),    # 21
  (172, 114, 82),   # 22
  (255, 127, 14),  # refrigerator
  (91, 163, 138),   # 24
  (153, 98, 156),   # 25
  (140, 153, 101),
  (158, 218, 229),  # shower curtain
  (100, 125, 154),
  (178, 127, 135),
  (120, 185, 128),
  (146, 111, 194),
  (44, 160, 44),  # toilet
  (112, 128, 144),  # sink
  (96, 207, 209),
  (227, 119, 194),  # bathtub
  (213, 92, 176),
  (94, 106, 211),
  (82, 84, 163),  # otherfurn
  (100, 85, 144)
])

arkitscenes_name_from_semantic_class_id = {
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    15: 'shelf',
    18: 'stove',
    19: 'washer',
    20: 'oven',
    21: 'dishwasher',
    22: 'fireplace',
    23: 'stool',
    24: 'refrigerator',
    25: 'tv_monitor',
    33: 'toilet',
    34: 'sink',
    36: 'bathtub'
}

arkitscenes_semantic_class_id_from_name = {
  'unlabeled': 0,  #
  'wall': 1,  #
  'floor': 2,  #
  'cabinet': 3,  #
  'bed': 4,  #
  'chair': 5,  #
  'sofa': 6,  #
  'table': 7,  #
  'door': 8,  #
  'window': 9,  #
  'bookshelf': 10,  #
  'picture': 11,  #
  'counter': 12,  #
  '~': 13,
  'desk': 14,  #
  'shelf': 15,
  'curtain': 16,  #
  '~': 17,
  'stove': 18,         #ARKIT
  'washer': 19,        #ARKIT
  'oven': 20,          #ARKIT
  'dishwasher': 21,    #ARKIT
  'fireplace': 22,     #ARKIT
  'stool': 23,         #ARKIT
  'refrigerator': 24,  #
  'tv_monitor': 25,
  '~': 26,
  '~': 27,
  'shower curtain': 28,  #
  '~': 29,
  '~': 30,
  '~': 31,
  '~': 32,
  'toilet': 33,
  'sink': 34,
  '~': 35,
  'bathtub': 36,
  '~': 37,
  '~': 38,
  'otherfurn': 39,
  '~': 40}


ARKITSCENES_SEMANTIC_CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'shelf', 'curtain', 'stove', 'washer', 'oven', 'dishwasher', 'fireplace', 'stool', 'refrigerator', 'tv_monitor', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
ARKITSCENES_INSTANCE_CLASS_LABELS =                  ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'shelf', 'curtain', 'stove', 'washer', 'oven', 'dishwasher', 'fireplace', 'stool', 'refrigerator', 'tv_monitor', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

# WARNING: those arrays are used within the network
# USED FOR DETECTION NETWORK
ARKITSCENES_SEMANTIC_VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 28, 33, 34, 36, 39])
ARKITSCENES_SEMANTIC_VALID_CLASS_IDS_torch = torch.Tensor(ARKITSCENES_SEMANTIC_VALID_CLASS_IDS)  # len = 20
ARKITSCENES_SEMANTIC_ID2IDX = torch.zeros(41).fill_(-100).long()
# Needed to map semantic ids to ones valid for scene segmentation (= valid classes W wall, ceiling, floor)
ARKITSCENES_SEMANTIC_ID2IDX[ARKITSCENES_SEMANTIC_VALID_CLASS_IDS] = torch.arange(len(ARKITSCENES_SEMANTIC_VALID_CLASS_IDS)).long()

# USED FOR REFINEMENT NETWORK
ARKITSCENES_INSTANCE_VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 28, 33, 34, 36, 39])  # len = 18
ARKITSCENES_INSTANCE_VALID_CLASS_IDS_torch = torch.Tensor(ARKITSCENES_INSTANCE_VALID_CLASS_IDS).long()
ARKITSCENES_INSTANCE_ID2IDX = torch.zeros(41).fill_(-100).long()
# Needed to map semantic ids to ones valid for instance segmentation (= valid classes WO wall, ceiling, floor)
ARKITSCENES_INSTANCE_ID2IDX[ARKITSCENES_INSTANCE_VALID_CLASS_IDS] = torch.arange(len(ARKITSCENES_INSTANCE_VALID_CLASS_IDS)).long()


def read_scene(path_ply, cfg):
    """Read the scene .ply.

    :return
        positions: 3D-float position of each vertex/point
        normals: 3D-float normal of each vertex/point (as computed by open3d)
        colors: 3D-float color of each vertex/point [0..1]
    """
    mesh = o3d.io.read_triangle_mesh(path_ply)

    # Apply geometric augmentation 
    if False and cfg.augmentation:
        # rotation around x,y,z
        if np.random.rand() < cfg.rotation_aug[0]:
            augmentation.rotate_mesh(mesh, max_xy_angle=cfg.rotation_aug[1], individual_prob=cfg.rotation_aug[2])

        # rotation around z (height) in 90 degree angles
        if cfg.rotation_90_aug:
            augmentation.rotate_mesh_90_degree(mesh)

        # mirroring / flipping
        if np.random.rand() < cfg.flipping_aug:
            Rt = np.eye(4)
            Rt[0][0] *= -1  # Randomly x-axis flip
            mesh.transform(Rt)

        if cfg.HAIS_jitter_aug:
            positions = np.asarray(mesh.vertices)
            positions -= positions.mean(0)
            m = np.eye(3)
            m += np.random.randn(3, 3) * 0.1
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])  # rotation
            positions = np.matmul(positions, m)
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # elastic distoriton
        positions = np.asarray(mesh.vertices)
        if np.random.rand() < cfg.elastic_distortion:
            for granularity, magnitude in augmentation.SCANNET_ELASTIC_DISTORT_PARAMS:
                positions = augmentation.elastic_distortion(positions, granularity, magnitude)
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # elastic distoriton HAIS setting
        if np.random.rand() < cfg.elastic_distortion_HAIS:
            positions = augmentation.HAIS_elastic(positions, 6 * (1 / cfg.voxel_size) // 50,
                                                  40 * (1 / cfg.voxel_size) / 50)
            positions = augmentation.HAIS_elastic(positions, 20 * (1 / cfg.voxel_size) // 50,
                                                  160 * (1 / cfg.voxel_size) / 50)
            positions -= positions.min(0)
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # random, independent noise on points
        if np.random.rand() < cfg.position_jittering[0]:
            displacements = cfg.position_jittering[1] * np.random.randn(*positions.shape)
            positions = positions + displacements
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # scaling
        if np.random.rand() < cfg.scaling_aug[0]:
            augmentation.scale_mesh(mesh, cfg.scaling_aug[1], cfg.scaling_aug[2])

    # positions = np.asarray(point_cloud.vertices)
    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    positions = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)


    # Color transformations
    if cfg.augmentation:
        # Contrast auto contrast
        if np.random.rand() < cfg.chromatic_auto_contrast:
            chromatic_auto_contrast = augmentation.ChromaticAutoContrast()
            colors = chromatic_auto_contrast(colors)

        # Chromatic translation
        if np.random.rand() < cfg.chromatic_translation[0]:
            trans_range_ratio = cfg.chromatic_translation[1]
            chromatic_translation = augmentation.ChromaticTranslation(trans_range_ratio)
            colors = chromatic_translation(colors)

        # Chromatic Jitter
        if np.random.rand() < cfg.color_jittering_aug[0]:
            colors = augmentation.color_jittering(colors, -cfg.color_jittering_aug[1], cfg.color_jittering_aug[1])

        # Random Brightness
        if np.random.rand() < cfg.random_brightness[0]:
            colors = augmentation.random_brightness(colors, cfg.random_brightness[1])

        # mix3d color augmentation (RandomBrightnessContrast, RGBShift) and normalization
        if cfg.mix_3d_color_aug:
            # color comes in, in [0,1] comes out in R (distributed around 0) (-> will break some visualizations)
            colors = augmentation.apply_mix3d_color_aug(colors)
        if cfg.apply_hue_aug:
            # color comes in, in [0,1] comes out in R (distributed around 0) (-> will break some visualizations)
            colors = augmentation.apply_hue_aug(colors)
    return positions, normals, colors

def is_foreground (sem):
    return sem > 2


def read_bounding_box_annotations(annotations_file, axis_aligned_bb=False):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    instances = annotations['data']  # list of dicts
    stats = annotations['stats']  # dict

    num_instances = len(instances)
    unique_instance_ids = np.array(range(num_instances))
    semantics = np.zeros_like(unique_instance_ids)
    bb_centers = np.zeros([num_instances, 3])
    bb_bounds = np.zeros([num_instances, 3])
    bb_rotations = np.zeros([num_instances, 9])
    for i, instance in enumerate(instances):
        # Semantic class
        semantic_class_name = instance['label']
        semantic_class_id = arkitscenes_semantic_class_id_from_name[semantic_class_name]
        semantics[i] = semantic_class_id
        # Bounding Box Annotation
        rotation = np.array(instance["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
        transform = np.array(instance["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        scale = np.array(instance["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)

        bb_centers[i, :] = transform
        bb_bounds[i, :] = scale * 0.5
        bb_rotations[i, :] = np.reshape(rotation, [1, 9])

        if axis_aligned_bb:
            def get_rotated_bounds(bb_bounds, rotation):
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
            bb_bounds[i, :] = get_rotated_bounds(bb_bounds[i, :], rotation)

    return unique_instance_ids, semantics, bb_centers, bb_bounds, bb_rotations


def process_scene(scene_name, mode, cfg, do_augmentations=False, visualize_for_debugging=False, subsample_rate=10):
    """Process scene: extracts ground truth labels (instance and semantics) and computes centers

    :return
        scene: dictionary containing
            positions: 3D-float position of each vertex/point
            normals: 3D-float normal of each vertex/point (as computed by open3d)
            colors: 3D-float color of each vertex/point [0..1SEMANTIC_VALID_CLASS_IDS]
        labels:  dictionary containing
            semantic_labels: N x 1 int32
            instance_labels: N x 1 int32
            centers: N x 3 float32
            center_distances: N x 1 float32
    """

    if mode == 'train':
        data_path = os.path.join(cfg.data_dir, '3dod/Training')
        segmentation_path = os.path.join(cfg.data_dir, '3dod/segmented_train_clean',
                                         f'{scene_name}_3dod_mesh.0.010000.segs.json')
    elif mode == 'val':
        data_path = os.path.join(cfg.data_dir, '3dod/Validation')
        segmentation_path = os.path.join(cfg.data_dir, '3dod/segmented_val_clean',
                                         f'{scene_name}_3dod_mesh.0.010000.segs.json')
    elif mode == 'predict_specific_scene':
        data_path = os.path.join(cfg.data_dir, '3dod/Training')
        segmentation_path = os.path.join(cfg.data_dir, '3dod/segmented_train_clean',
                                         f'{scene_name}_3dod_mesh.0.010000.segs.json')
        if not os.path.exists (os.path.join (data_path, scene_name, f'{scene_name}_3dod_mesh.ply')):
            data_path = os.path.join(cfg.data_dir, '3dod/Validation')
            segmentation_path = os.path.join(cfg.data_dir, '3dod/segmented_val_clean',
                                         f'{scene_name}_3dod_mesh.0.010000.segs.json')
    else:
        print(f'Unknown mode: {mode} for ARKitScenes dataset')
        exit(0)

    # Setup pathes to all necessary files
    path_ply = os.path.join(data_path, scene_name, f'{scene_name}_3dod_mesh.ply')
    path_annotations = os.path.join(data_path, scene_name, f'{scene_name}_3dod_annotation.json')

    # ----------------- INPUT SCENE ----------------------------#
    # Read point clouds, extract semantic & instance labels, compute centers

    positions, normals, colors = read_scene(path_ply, cfg)

    with open(segmentation_path) as f:
         per_point_segment_ids = json.load(f)
    per_point_segment_ids = np.asarray(per_point_segment_ids["segIndices"], dtype='int32')

    scene = {'name': scene_name,
             'positions': positions[::subsample_rate, :],
             'normals': normals[::subsample_rate, :],
             'colors': colors[::subsample_rate, :],
             'segments': per_point_segment_ids[::subsample_rate]}

    if mode == 'test':
        return scene, None

    unique_instances, per_instance_semantics, per_instance_bb_centers, per_instance_bb_bounds, per_instance_bb_rotations \
        = read_bounding_box_annotations(path_annotations)

    translation_xy = np.mean(scene['positions'][:, 0:2], 0)
    translation_z = np.min(scene['positions'][:, 2])
    scene['positions'][:, 0:2] -= translation_xy
    scene['positions'][:, 2] -= translation_z
    for i, bb_center in enumerate(per_instance_bb_centers):
        per_instance_bb_centers[i][0:2] -= translation_xy
        per_instance_bb_centers[i][2] -= translation_z

    if cfg.augmentation:
        # rotation around x,y,z
        individual_prob = cfg.rotation_aug[2]
        max_xy_angle = cfg.rotation_aug[1]
        if np.random.rand() < cfg.rotation_aug[0]:
            random_z_angle = 0
            random_x_angle = 0
            random_y_angle = 0
            if random.random() < individual_prob:
                random_z_angle = np.random.uniform(0, np.pi * 2)
            if random.random() < individual_prob:
                random_x_angle = np.random.uniform(-max_xy_angle, max_xy_angle)
            if random.random() < individual_prob:
                random_y_angle = np.random.uniform(-max_xy_angle, max_xy_angle)
            rotation_matrix = R.from_euler('xyz', [random_x_angle, random_y_angle, random_z_angle]).as_matrix()
            scene['positions'] = (rotation_matrix @ scene['positions'].T).T
            scene['normals'] = (rotation_matrix @ scene['normals'].T).T
            per_instance_bb_centers = (rotation_matrix @ per_instance_bb_centers.T).T
            for i in range(per_instance_bb_rotations.shape[0]):
                rot = np.reshape(per_instance_bb_rotations[i, :], [3, 3])
                rot = rotation_matrix.T @ rot
                per_instance_bb_rotations[i, :] = np.reshape(rot, [9])

        # random, independent noise on points
        if np.random.rand() < cfg.position_jittering[0]:
            scene['positions'] += cfg.position_jittering[1] * np.random.randn(*scene['positions'].shape)

        # scaling_aug = [probability, min_scale, max_scale]
        if np.random.rand() < cfg.scaling_aug[0]:
            scale = np.random.uniform(cfg.scaling_aug[1], cfg.scaling_aug[2])
            scene['positions'] *= scale
            for i, bb_center in enumerate(per_instance_bb_centers):
                per_instance_bb_centers[i] *= scale
                per_instance_bb_bounds[i] *= scale

    if visualize_for_debugging:
        v = viz.Visualizer()
        mask = scene['positions'][:, 2] < 10000.5
        v.add_points('RGB', scene['positions'][mask], (scene['colors'][mask] * 255).astype(int), scene['normals'][mask],
                     point_size=25, visible=True)
        for i in unique_instances.tolist():
            rotation_matrix = np.reshape(per_instance_bb_rotations[i], [3, 3]).T
            rotation_quaternion = q.from_rotation_matrix(rotation_matrix)
            v.add_bounding_box(f'bb_{i}', per_instance_bb_centers[i, :],
                               per_instance_bb_bounds[i, :] * 2,
                               orientation=np.array(q.as_float_array(rotation_quaternion)).tolist(),
                               color=arkitscenes_color_map[per_instance_semantics[i]])
        v.save(f'viz_arkit;{scene_name}')

    labels = {'unique_instances': unique_instances,
              'per_instance_semantics': per_instance_semantics,
              'per_instance_bb_centers': per_instance_bb_centers,
              'per_instance_bb_bounds': per_instance_bb_bounds,
              'per_instance_bb_rotations': per_instance_bb_rotations}
    return scene, labels


if __name__ == '__main__':

    for key, value in arkitscenes_name_from_semantic_class_id.items():
        r = str(arkitscenes_color_map[key][0] / 255.0)[:6]
        g = str(arkitscenes_color_map[key][1] / 255.0)[:6]
        b = str(arkitscenes_color_map[key][2] / 255.0)[:6]
        print('\definecolor{'+value+'}{rgb}{'+r+','+g+','+b+'}')

    for key, value in arkitscenes_name_from_semantic_class_id.items():
        print('\\textcolor{'+value+'}{\\ColorMapCircle}\,'+str.capitalize(value))

    import config_loader as cfg_loader
    cfg = cfg_loader.get_config()
    path = os.path.join(cfg.data_dir, '3dod/Training')
    for scene_name in os.listdir(path)[0:15]:
        mode = 'train'
        process_scene(scene_name, mode, cfg, do_augmentations=True, visualize_for_debugging=True)
