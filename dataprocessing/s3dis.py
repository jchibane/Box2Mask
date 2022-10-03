import sys
sys.path.append('.')

import open3d as o3d
import numpy as np
import pyviz3d.visualizer as viz
import glob
from natsort import natsorted
import os
import json
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree as KDTree
import time, math
import dataprocessing.augmentation as augmentation
import torch

def generate_color_map (max_ids):
    mapping = [[np.random.randint (255), np.random.randint (255), np.random.randint (255)] for _ in range (max_ids)]
    return np.array (mapping)

ID2NAME = {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookshelf', 11:'board', 12:'clutter'}
ID2NAME = [ID2NAME [i] for i in range (13)]
S3DIS_SEMANTICS_COLORS = np.array (
    [(174, 199, 232),  # ceiling
    (152, 223, 138),  # floor
    (31, 119, 180),   # wall
    (255, 187, 120),  # column
    (188, 189, 34),   # beam
    (140, 86, 75),    # window
    (255, 152, 150),  # door
    (214, 39, 40),    # table
    (197, 176, 213),  # chair
    (148, 103, 189),  # bookcase
    (196, 156, 148),  # sofa
    (23, 190, 207),   # board
    (178, 76, 76),]   # clutter
)

# WARNING: those arrays are used within the network
S3DIS_SEMANTIC_VALID_CLASS_IDS = np.array(range (13))
S3DIS_SEMANTIC_VALID_CLASS_IDS_torch = torch.Tensor(S3DIS_SEMANTIC_VALID_CLASS_IDS) 
S3DIS_INSTANCE_VALID_CLASS_IDS = np.array(range (13))

S3DIS_INSTANCE_VALID_CLASS_IDS_torch = torch.Tensor(S3DIS_INSTANCE_VALID_CLASS_IDS).long()
S3DIS_INSTANCE_ID2IDX = torch.zeros(13).fill_(-1).long()
S3DIS_INSTANCE_ID2IDX[S3DIS_INSTANCE_VALID_CLASS_IDS] = torch.arange(len(S3DIS_INSTANCE_VALID_CLASS_IDS)).long()

S3DIS_SEMANTIC_ID2IDX = torch.zeros(300).fill_(-100).long()
# Needed to map semantic ids to ones valid for scene segmentation (= valid classes W wall, ceiling, floor)
S3DIS_SEMANTIC_ID2IDX[S3DIS_SEMANTIC_VALID_CLASS_IDS] = torch.arange(len(S3DIS_SEMANTIC_VALID_CLASS_IDS)).long()

def get_scene_names (mode, cfg):
    scene_npy_pths = glob.glob (os.path.join (cfg.data_dir, 'Area_*/*.npy'))
    scene_names = [pth.split ('/')[-2] + '.' + pth.split ('/')[-1].split ('.')[0] for pth in scene_npy_pths]

    if mode == "train":
        valid_set_prefix = "Area_" + str (cfg.s3dis_split_fold)
        scene_names = [name for name in scene_names if valid_set_prefix not in name]
    if mode == 'val':
        valid_set_prefix = "Area_" + str (cfg.s3dis_split_fold)
        scene_names = [name for name in scene_names if valid_set_prefix in name]
    
    return scene_names

def refine_segments (segments, counts, positions, minVerts=20):
    """ merge too small segments to large nearby segment
    """
    segcount_per_point = counts [segments]
    large_enough = segcount_per_point > minVerts
    too_small = segcount_per_point <= minVerts
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(positions[large_enough])
    dist, qualified_2_disqualified = nbrs.kneighbors(positions[too_small])
    disqualified2qualified = qualified_2_disqualified[:,0]
    segments [too_small] = segments[large_enough] [disqualified2qualified]
    
    return segments

def semantics_to_forground_mask (semantics, cfg=None):
    if cfg.ignore_wall_ceiling_floor:
        return semantics > 2
    return semantics >= 0
    
def is_foreground (sem):
    return sem > 2

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
        bb_center_distances = np.linalg.norm(offsets, axis=1)
        per_point_bb_center_distances[instance_mask] = bb_center_distances.reshape((-1,1))

        # bb radius
        radius = np.max(bb_center_distances).reshape((-1,1))
        per_point_bb_radius[instance_mask] = radius
        per_instance_bb_radius[i] = radius

    return per_point_bb_centers, per_point_bb_offsets, per_point_bb_bounds, \
           per_point_bb_center_distances, per_point_bb_radius, \
           instances, per_instance_semantics, per_instance_bb_centers, per_instance_bb_bounds, per_instance_bb_radius

def seg2label (segments, label_ids):
    # Use major voting to assign label for each segment
    unique_segments_ids = np.unique(segments)
    seg2labelID = np.zeros(np.max(unique_segments_ids) + 1, dtype='int32')
    seg2labelID.fill(np.inf)
    for seg_id in unique_segments_ids:
        seg_mask = segments == seg_id
        
        seg_label_ids = label_ids[seg_mask]
        counts = np.bincount (seg_label_ids)
        most_frequent_labels = np.argmax(counts)
        
        seg2labelID[seg_id] = most_frequent_labels
    per_point_segment_labelID = seg2labelID [segments]
    return per_point_segment_labelID, seg2labelID

def read_scene_from_numpy (scene_name, cfg, do_augmentations=False):
    """read_scene_from_numpy: read scene informationfrom numpy

    :return
        scene: dictionary containing
            name: name of the scene informat [area].[place]
            positions: 3D-float position of each vertex/point
            normals: 3D-float normal of each vertex/point (as computed by open3d) 
            colors: 3D-float color of each vertex/point [0..1]
            segments: segments id of each vertex/point: N x 1 int32
        labels:  dictionary containing
            semantic_labels: N x 1 int32
            instance_labels: N x 1 int32
            centers: N x 3 float32
            center_distances: N x 1 float32
    """
    scene_npy_path = os.path.join (cfg.data_dir, scene_name.split ('.') [0] + '/' + scene_name [len("Area_*") + 1:] + '.normals.instance.npy')
    data = np.load (scene_npy_path)
    
    positions = data [:,:3].astype (np.float32)
    colors = data [:,3:6].astype (np.float) / 255
    positions = positions - positions.mean (0)
    positions[:, 2] -= np.min (positions [:, 2])
    normals = data [:,6:9].astype (np.float)
    semantics = data [:, -2].astype (np.int32)
    instances = data [:, -1].astype (np.int32)

    # Basic augmentations (rotation, scaling, flipping x-y)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype (np.float32))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype (np.float32))

    if cfg and cfg.augmentation and do_augmentations:
        # rotation around x,y,z
        if np.random.rand () < cfg.rotation_aug[0]:
            augmentation.rotate_mesh (pcd)
        if np.random.rand () < cfg.scaling_aug [0]:
            augmentation.scale_mesh (pcd, cfg.scaling_aug[1], cfg.scaling_aug[2])
        # rotation around z (height) in 90 degree angles
        if cfg.rotation_90_aug:
            augmentation.rotate_mesh_90_degree(pcd)
        if np.random.rand () < cfg.flipping_aug:
            Rt = np.eye (4)
            Rt [0][0] *= -1 # Randomly x-axis flip
            pcd.transform (Rt)
        if np.random.rand () < cfg.flipping_aug:
            Rt = np.eye (4)
            Rt [1][1] *= - 1 # Randomly y-axis flip
            pcd.transform (Rt)
    
    positions = np.asarray(pcd.points)
    normals = np.asarray (pcd.normals)

    # Apply geometric augmentation
    if do_augmentations and cfg.augmentation:
        if np.random.rand () < cfg.elastic_distortion:
            elastic_distortion = augmentation.ElasticDistortion ()
            positions = elastic_distortion (positions)
            pcd.points = o3d.utility.Vector3dVector(positions)

        # elastic distoriton HAIS setting
        if np.random.rand () < cfg.elastic_distortion_HAIS:
            positions -= positions.mean (0)
            positions = augmentation.HAIS_elastic(positions, 6 * (1/cfg.voxel_size) // 50, 40 * (1/cfg.voxel_size) / 50)
            positions = augmentation.HAIS_elastic(positions, 20 * (1/cfg.voxel_size) // 50, 160 * (1/cfg.voxel_size) / 50)
            positions -= positions.mean (0)
            positions[:, 2] -= np.min (positions [:, 2])
            pcd.points = o3d.utility.Vector3dVector(positions)
        
        if np.random.rand () < cfg.position_jittering [0]:
            displacements = cfg.position_jittering [1] * np.random.randn (*positions.shape)
            positions = positions + displacements
            pcd.points = o3d.utility.Vector3dVector(positions)

        if cfg.HAIS_jitter_aug:
            positions -= positions.mean(0)
            pcd.points = o3d.utility.Vector3dVector(positions)
            Rt = np.eye (4)
            m = np.eye(3)
            m += np.random.randn(3, 3) * 0.1
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
            Rt[:3,:3] = m
            pcd.transform (Rt)
            positions = np.asarray(pcd.points)
            positions[:, 2] -= np.min (positions [:, 2])
            pcd.points = o3d.utility.Vector3dVector(positions)

    # Color transformations
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

        if cfg.apply_hue_aug:
            colors = augmentation.apply_hue_aug(colors)


    positions = np.asarray(pcd.points)
    normals = np.asarray (pcd.normals)

    if cfg.superpoint_algo == 'learned_superpoint':
        PATH_SEGMENTS_LABELS_INFO = os.path.join (cfg.data_dir, 'segment_labels/learned_superpoint_graph_segmentations/')

    path_scene_segments_labels_info = PATH_SEGMENTS_LABELS_INFO + '/' + scene_name + '.npy'
    segments_labels_info = np.load (path_scene_segments_labels_info, allow_pickle=True).item ()
    segments = segments_labels_info ['segments']
    per_point_segment_instanceID = segments_labels_info ['per_point_segment_instanceID']
    seg2instanceID = segments_labels_info ['seg2instanceID']
    per_point_segment_semanticID = segments_labels_info ['per_point_segment_semanticID']
    seg2semanticID = segments_labels_info ['seg2semanticID']

    if cfg.point_sampling_rate is not None:
        num_scene_points = len (positions)
        sampling_mask = np.zeros (num_scene_points, dtype=np.bool)
        if not do_augmentations:
            # During evaluation or testing, sample every 4 points
            sampling_point_ids = np.array (range (num_scene_points)) [::4]
        else:
            # During training, sample points randomly with an user input sampling rate
            sampling_point_ids = np.random.choice (range (num_scene_points), int (num_scene_points * cfg.point_sampling_rate), replace=False)
        sampling_mask [sampling_point_ids] = True

        segments = segments [sampling_mask]

        # Remaping contiguous segments ID
        per_point_segment_semanticID = per_point_segment_semanticID [sampling_mask]
        per_point_segment_instanceID = per_point_segment_instanceID [sampling_mask]
        positions = positions [sampling_mask]
        colors = colors [sampling_mask]
        normals = normals [sampling_mask]
        instances = instances [sampling_mask]
        semantics = semantics [sampling_mask]
               
    scene = {
        'name': scene_name,
        'positions': positions,
        'colors': colors,
        'normals': normals,
        'segments': segments,
    }
    labels = {
        'instances': instances,
        'semantics': semantics,
        'per_point_segment_instanceID': per_point_segment_instanceID,
        'per_point_segment_semanticID': per_point_segment_semanticID,
        'seg2instanceID': seg2instanceID,
        'seg2semanticID': seg2semanticID,
    }
    return scene, labels

def process_scene(scene_name, mode, configuration, do_augmentations=False):
    """Process scene: extracts ground truth labels (instance and semantics) and computes centers

    :return
        scene: dictionary containing
            positions: 3D-float position of each vertex/point
            normals: 3D-float normal of each vertex/point (as computed by open3d)
            colors: 3D-float color of each vertex/point [0..1]
        labels:  dictionary containing
            semantic_labels: N x 1 int32
            instance_labels: N x 1 int32
            centers: N x 3 float32
            center_distances: N x 1 float32
    """
    cfg = configuration

    # Read point clouds, extract semantic & instance labels, compute centers
    scene, labels = read_scene_from_numpy(scene_name, configuration, do_augmentations=do_augmentations)
    centers, center_distances = compute_avg_centers(scene ['positions'], labels ['instances'])
    
    bb_centers, bb_offsets, bb_bounds, bb_center_distances, bb_radius, \
        unique_instances, per_instance_semantics, per_instance_bb_centers, per_instance_bb_bounds, per_instance_bb_radius \
        = compute_bounding_box(scene ['positions'], labels['instances'], labels['semantics'])

    # make sure the unique instance ids can be used as array indices for 'per_instance_XX'
    assert np.all(unique_instances == range(len(unique_instances)))

    labels ['per_instance_bb_radius'] = per_instance_bb_radius
    labels ['per_instance_bb_bounds'] = per_instance_bb_bounds
    labels ['per_instance_bb_centers'] = per_instance_bb_centers
    labels ['per_instance_semantics'] = per_instance_semantics
    labels ['unique_instances'] = unique_instances
    labels ['bb_radius'] = bb_radius
    labels ['bb_center_distances'] = bb_center_distances
    labels ['seg2inst'] = labels ['seg2instanceID']
    labels ['bb_bounds'] = bb_bounds
    labels ['bb_offsets'] = bb_offsets
    labels ['bb_centers'] = bb_centers
    labels ['center_distances'] = center_distances
    labels ['centers'] = centers

    return scene, labels
