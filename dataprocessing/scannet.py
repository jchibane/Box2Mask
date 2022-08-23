"""Prepare scannet data. Read the ground truth instance and semantic labels, and compute centers."""

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

# for visulaization with pyviz3d
scannet_color_map = np.array([
  (0, 0, 0), # unlabeled
  (174, 199, 232),  # wall
  (152, 223, 138),  # floor
  (31, 119, 180),  # cabinet
  (255, 187, 120),  # bed
  (188, 189, 34),  # chair
  (140, 86, 75),  # sofa
  (255, 152, 150),  # table
  (214, 39, 40),  # door
  (197, 176, 213),  # window
  (148, 103, 189),  # bookshelf
  (196, 156, 148),  # picture
  (23, 190, 207),  # counter
  (178, 76, 76),
  (247, 182, 210),  # desk
  (66, 188, 102),
  (219, 219, 141),  # curtain
  (140, 57, 197),
  (202, 185, 52),
  (51, 176, 203),
  (200, 54, 131),
  (92, 193, 61),
  (78, 71, 183),
  (172, 114, 82),
  (255, 127, 14),  # refrigerator
  (91, 163, 138),
  (153, 98, 156),
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

scannet_class_names = [
  'unlabeled', #
  'wall',  #
  'floor',  #
  'cabinet',  #
  'bed',  #
  'chair',  #
  'sofa',  #
  'table',  #
  'door',  #
  'window',  #
  'bookshelf',  #
  'picture',  #
  'counter',  #
  '~',
  'desk',  #
  '~',
  'curtain',  #
  '~',
  '~',
  '~',
  '~',
  '~',
  '~',
  '~',
  'refrigerator',  #
  '~',
  '~',
  '~',
  'shower curtain',  #
  '~',
  '~',
  '~',
  '~',
  'toilet',
  'sink',
  '~',
  'bathtub',
  '~',
  '~',
  'otherfurn',
  '~']

SEMANTIC_CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
INSTANCE_CLASS_LABELS =                  ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

# WARNING: those arrays are used within the network
# USED FOR DETECTION NETWORK
SCANNET_SEMANTIC_VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]) # len = 20
SCANNET_SEMANTIC_VALID_CLASS_IDS_torch = torch.Tensor(SCANNET_SEMANTIC_VALID_CLASS_IDS) # len = 20
SCANNET_SEMANTIC_ID2IDX = torch.zeros(41).fill_(-100).long()
# Needed to map semantic ids to ones valid for scene segmentation (= valid classes W wall, ceiling, floor)
SCANNET_SEMANTIC_ID2IDX[SCANNET_SEMANTIC_VALID_CLASS_IDS] = torch.arange(len(SCANNET_SEMANTIC_VALID_CLASS_IDS)).long()

SCANNET_INSTANCE_VALID_CLASS_IDS =       np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]) # len = 18
SCANNET_INSTANCE_VALID_CLASS_IDS_torch = torch.Tensor(SCANNET_INSTANCE_VALID_CLASS_IDS).long()

SCANNET_INSTANCE_ID2IDX = torch.zeros(41).fill_(-100).long()
# Needed to map semantic ids to ones valid for instance segmentation (= valid classes WO wall, ceiling, floor)
SCANNET_INSTANCE_ID2IDX[SCANNET_INSTANCE_VALID_CLASS_IDS] = torch.arange(len(SCANNET_INSTANCE_VALID_CLASS_IDS)).long()

def remap_20_to_40(arr):
    ret_shape = list(arr.shape)
    ret_shape [1] = 41
    ret = np.zeros(ret_shape, dtype=arr.dtype)
    for i in range(20):
        ret[:, SEMANTIC_VALID_CLASS_IDS[i]] = arr[:, i]
    return ret

def is_foreground(sem):
    return (sem > 2) & (sem != 22)


def read_scene(path_ply, path_txt, cfg, align=False, do_augmentations=False, rotate_z=None):
    """Read the scene .ply.

    :return
        positions: 3D-float position of each vertex/point
        normals: 3D-float normal of each vertex/point (as computed by open3d)
        colors: 3D-float color of each vertex/point [0..1]
    """
    mesh = o3d.io.read_triangle_mesh(path_ply)
    if align:
        with open(path_txt) as f:
            lines = f.readlines()
        axis_alignment = ''
        for line in lines:
            if line.startswith('axisAlignment'):
                axis_alignment = line
                break
        if axis_alignment == '':
            raise ValueError('No axis alignment found!')
        Rt = np.array([float(v) for v in axis_alignment.split('=')[1].strip().split(' ')]).reshape([4, 4])
        mesh.transform(Rt)
    
    # Apply geometric augmentation
    if do_augmentations and cfg.augmentation:
        # rotation around x,y,z
        if np.random.rand () < cfg.rotation_aug[0]:
            augmentation.rotate_mesh (mesh, max_xy_angle = cfg.rotation_aug[1], individual_prob=cfg.rotation_aug[2])

        # rotation around z (height) in 90 degree angles
        if cfg.rotation_90_aug:
            augmentation.rotate_mesh_90_degree(mesh)

        # mirroring / flipping
        if np.random.rand () < cfg.flipping_aug:
            Rt = np.eye (4)
            Rt [0][0] *= -1 # Randomly x-axis flip
            mesh.transform(Rt)

        if cfg.HAIS_jitter_aug:
            positions = np.asarray(mesh.vertices)
            positions -= positions.mean(0)
            m = np.eye(3)
            m += np.random.randn(3, 3) * 0.1
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
            positions = np.matmul(positions, m)
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # elastic distoriton
        positions = np.asarray(mesh.vertices)
        if np.random.rand () < cfg.elastic_distortion:
            for granularity, magnitude in augmentation.SCANNET_ELASTIC_DISTORT_PARAMS:
                positions = augmentation.elastic_distortion(positions, granularity, magnitude)
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # elastic distoriton HAIS setting
        if np.random.rand () < cfg.elastic_distortion_HAIS:
            positions = augmentation.HAIS_elastic(positions, 6 * (1/cfg.voxel_size) // 50, 40 * (1/cfg.voxel_size) / 50)
            positions = augmentation.HAIS_elastic(positions, 20 * (1/cfg.voxel_size) // 50, 160 * (1/cfg.voxel_size) / 50)
            positions -= positions.min(0)
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # random, independent noise on points
        if np.random.rand () < cfg.position_jittering [0]:
            displacements = cfg.position_jittering [1] * np.random.randn (*positions.shape)
            positions = positions + displacements
            mesh.vertices = o3d.utility.Vector3dVector(positions)

        # scaling
        if np.random.rand () < cfg.scaling_aug [0]:
            augmentation.scale_mesh (mesh, cfg.scaling_aug[1], cfg.scaling_aug[2])

    elif rotate_z:
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, rotate_z/2))) 

    positions = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    normals = np.asarray(mesh.vertex_normals)

    # Color transformations
    colors = np.asarray(mesh.vertex_colors)
    if do_augmentations and cfg.augmentation:
        # Contrast auto contrast
        if np.random.rand () < cfg.chromatic_auto_contrast:
            chromatic_auto_contrast = augmentation.ChromaticAutoContrast ()
            colors = chromatic_auto_contrast (colors)

        # Chromatic translation
        if np.random.rand () < cfg.chromatic_translation [0]:
            trans_range_ratio = cfg.chromatic_translation [1]
            chromatic_translation = augmentation.ChromaticTranslation (trans_range_ratio)
            colors = chromatic_translation (colors)

        # Chromatic Jitter
        if np.random.rand () < cfg.color_jittering_aug [0]:
            colors = augmentation.color_jittering (colors, -cfg.color_jittering_aug [1], cfg.color_jittering_aug [1])

        # Random Brightness
        if np.random.rand () < cfg.random_brightness [0]:
            colors = augmentation.random_brightness (colors, cfg.random_brightness [1])

        # mix3d color augmentation (RandomBrightnessContrast, RGBShift) and normalization
        if cfg.mix_3d_color_aug:
            # color comes in, in [0,1] comes out in R (distributed around 0) (-> will break some visualizations)
            colors = augmentation.apply_mix3d_color_aug(colors)
        if cfg.apply_hue_aug:
            # color comes in, in [0,1] comes out in R (distributed around 0) (-> will break some visualizations)
            colors = augmentation.apply_hue_aug(colors)
    return positions, normals, colors

def read_labels(label_map_file, path_aggregation, per_point_segment_ids):

    # Create label map, i.e. map from label_name to label_id
    label_map = {}
    with open(label_map_file, 'r') as f:
        lines = csv.reader(f, delimiter='\t')
        cnt = 0
        for line in lines:
            if cnt > 0:
                if len(line[4]) > 0:
                    label_map[line[1]] = line[4]
                else:
                    label_map[line[1]] = '0'
            cnt += 1

    # Read semantic labels
    with open(path_aggregation) as f:
        aggregation_data = json.load(f)

    # semantics and instances are overwritten with non-zero values in the following
    # if 0 values are encountered this means the point had no annotation - i.e. zero is our default here
    per_point_semantic_labels = np.zeros((len(per_point_segment_ids)), dtype='int32')
    per_point_instance_labels = np.zeros((len(per_point_segment_ids)), dtype='int32')

    for instance_id, instance in enumerate(aggregation_data["segGroups"]):  
        semantic_string = instance["label"]
        for segment in instance["segments"]:
            ind = per_point_segment_ids == int(segment)
            if semantic_string in label_map:
                semantic_id = label_map[semantic_string]
            else:
                semantic_id = '-'
            per_point_semantic_labels[ind] = int(semantic_id)
            per_point_instance_labels[ind] = instance_id + 1

    # There are some buggy scenes (like e.g. scene0217_00) that double define instances
    # here I handle this bug:
    unique_instance_ids = np.unique(per_point_instance_labels)
    if not np.all(unique_instance_ids == range(len(unique_instance_ids))):
        per_point_instance_labels = rankdata(per_point_instance_labels, method='dense') - 1  # (num_voxels_in_batch)

    # create vectorized seg_id to instance mapping
    unique_segments_ids = np.unique(per_point_segment_ids)
    seg2inst = np.zeros(np.max(unique_segments_ids) + 1, dtype='int32')
    seg2inst.fill(np.inf)
    for seg_id in unique_segments_ids:
        seg_mask = per_point_segment_ids == seg_id
        assert len(np.unique(per_point_instance_labels[seg_mask])) == 1
        inst_id = per_point_instance_labels[seg_mask][0]
        seg2inst[seg_id] = inst_id

    return per_point_semantic_labels, per_point_instance_labels, seg2inst


def compute_avg_centers(positions, instance_labels):
    per_point_centers = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_offsets = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_center_distances = np.zeros((instance_labels.shape[0], 1), dtype='float32')

    for instance_id in set(instance_labels):
        instance_mask = (instance_id == instance_labels)

        # compute AVG centers
        instance_center = np.mean(positions[instance_mask], axis=0)
        per_point_centers[instance_mask] = instance_center
        per_point_offsets[instance_mask] = per_point_centers[instance_mask] - positions[instance_mask]
        per_point_center_distances = np.linalg.norm(per_point_offsets, axis=1)

    return per_point_centers, per_point_center_distances


def compute_bounding_box(positions, instance_labels, semantic_labels):
    per_point_bb_centers = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_bb_offsets = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_bb_bounds = np.zeros((instance_labels.shape[0], 3), dtype='float32')
    per_point_bb_center_distances = np.zeros((instance_labels.shape[0], 1), dtype='float32')
    per_point_bb_radius = np.zeros((instance_labels.shape[0], 1), dtype='float32')

    instances = np.unique(instance_labels)
    per_instance_semantics = np.zeros((len(instances)),  dtype='int32')
    per_instance_bb_centers = np.zeros((len(instances), 3), dtype='float32')
    per_instance_bb_bounds = np.zeros((len(instances), 3), dtype='float32')
    per_instance_bb_radius = np.zeros((len(instances)), dtype='float32')

    for i, instance_id in enumerate(instances):
        instance_mask = (instance_id == instance_labels)
        instance_points = positions[instance_mask]
        per_instance_semantics[i] = semantic_labels[instance_mask][0]

        # bb center
        max_bounds = np.max(instance_points, axis=0)
        min_bounds = np.min(instance_points, axis=0)
        bb_center = (min_bounds + max_bounds) / 2
        per_point_bb_centers[instance_mask] = bb_center
        per_instance_bb_centers[i] = bb_center

        # bb bounds
        bb_bounds = max_bounds - bb_center
        per_point_bb_bounds[instance_mask] = bb_bounds
        per_instance_bb_bounds[i] = bb_bounds

        # bb center offsets
        offsets = bb_center - instance_points
        per_point_bb_offsets[instance_mask] =  offsets

        # bb center distances
        # import ipdb; ipdb.set_trace()
        bb_center_distances = np.linalg.norm(offsets, axis=1)
        per_point_bb_center_distances[instance_mask] = bb_center_distances.reshape((-1,1))

        # bb radius
        radius = np.max(bb_center_distances).reshape((-1,1))
        per_point_bb_radius[instance_mask] = radius
        per_instance_bb_radius[i] = radius

    return per_point_bb_centers, per_point_bb_offsets, per_point_bb_bounds, \
           per_point_bb_center_distances, per_point_bb_radius, \
           instances, per_instance_semantics, per_instance_bb_centers, per_instance_bb_bounds, per_instance_bb_radius

def process_scene(scene_name, mode, configuration, do_augmentations=False, rotate_z = None):
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

    global cfg
    cfg = configuration

    align = cfg.align
    if not mode == 'test':
        data_path = os.path.join(cfg.data_dir , 'scans')
        path_segmention = os.path.join(data_path, scene_name, f'{scene_name}_vh_clean_2.0.010000.segs.json')
    else:
        align = False
        data_path = os.path.join(cfg.data_dir , 'scans_test')
        path_segmention = os.path.join(cfg.data_dir, 'scans_test_segmented', f'{scene_name}_vh_clean_2.0.010000.segs.json')

    # Setup pathes to all necessary files
    path_txt = os.path.join(data_path, scene_name, f'{scene_name}.txt')
    path_ply = os.path.join(data_path, scene_name, f'{scene_name}_vh_clean_2.ply')
    path_aggregation = os.path.join(data_path, scene_name, f'{scene_name}.aggregation.json')
    label_map_file = os.path.join(data_path, '../scannetv2-labels.combined.tsv')


    # ----------------- INPUT SCENE ----------------------------#
    # Read point clouds, extract semantic & instance labels, compute centers

    positions, normals, colors = read_scene(path_ply, path_txt, configuration, align=align, do_augmentations=do_augmentations, rotate_z = rotate_z)

    with open(path_segmention) as f:
        per_point_segment_ids = json.load(f)
    per_point_segment_ids = np.asarray(per_point_segment_ids["segIndices"], dtype='int32')

    scene = {'name': scene_name, 'positions': positions, 'normals': normals, 'colors': colors,
             'segments': per_point_segment_ids}

    if mode == 'test':
        return scene, None

    # ----------------- GT LABELS  ----------------------------#
    semantic_labels, instance_labels, seg2inst = read_labels(label_map_file, path_aggregation, per_point_segment_ids)

    centers, center_distances = compute_avg_centers(positions, instance_labels)

    bb_centers, bb_offsets, bb_bounds, bb_center_distances, bb_radius, \
    unique_instances, per_instance_semantics, per_instance_bb_centers, per_instance_bb_bounds, per_instance_bb_radius \
    = compute_bounding_box(positions, instance_labels, semantic_labels)

    # make sure the unique instance ids can be used as array indices for 'per_instance_XX'
    assert np.all(unique_instances == range(len(unique_instances)))
    
    labels = {'semantics': semantic_labels, 'instances': instance_labels,
              'centers': centers, 'center_distances': center_distances,
              'bb_centers':bb_centers, 'bb_offsets':bb_offsets, 'bb_bounds':bb_bounds, 'seg2inst': seg2inst,
              'bb_center_distances': bb_center_distances, 'bb_radius':bb_radius,
              'unique_instances':unique_instances, 'per_instance_semantics':per_instance_semantics,
              'per_instance_bb_centers':per_instance_bb_centers, 'per_instance_bb_bounds': per_instance_bb_bounds,
              'per_instance_bb_radius': per_instance_bb_radius }


    return scene, labels
