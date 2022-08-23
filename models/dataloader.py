from __future__ import division
from operator import is_

from numpy.lib.type_check import _is_type_dispatcher
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import traceback
import copy

import dataprocessing.scannet as scannet
import dataprocessing.arkitscenes as arkitscenes
import MinkowskiEngine as ME
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
from utils.util import to_unique, is_within_bb_np as is_within_bb
import scipy.stats as stats

import random

class ScanNet(Dataset):

    def __init__(self, mode, cfg, do_augmentations=True):

        self.do_augmentations = do_augmentations
        self.mode = mode
        self.cfg = cfg

        if mode == 'train+val':
            train_list = np.load(cfg.data_split, allow_pickle=True)['train']
            val_list = np.load(cfg.data_split, allow_pickle=True)['val']
            self.data_list = np.concatenate((train_list, val_list))
        elif mode == 'predict_specific_scene':
            self.data_list = [cfg.predict_specific_scene,]
        else:
            self.data_list = np.load(cfg.data_split, allow_pickle=True)[mode]

        # for testing purposes: over-fit a model to a single scene, via an integer index
        if cfg.overfit_to_single_scene is not None:
            self.data_list = [self.data_list[cfg.overfit_to_single_scene]] * 100

        # for testing purposes: over-fit a model to a single scene, via a scene name string
        if cfg.overfit_to_single_scene_str is not None:
            self.data_list = [cfg.overfit_to_single_scene_str,] * 100
        if cfg.dataset_size is not None:
            self.data_list = self.data_list[:cfg.dataset_size]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # will store all return data of this method
        ret = {}
        scene_name = self.data_list[idx]

        scene, labels = scannet.process_scene(scene_name, self.mode, self.cfg, do_augmentations=self.do_augmentations)

        #---------------- Voxelization Code (START) ----------------- #
        # Translate scene to avoid negative coords
        input_coords = scene["positions"] - min(0, np.min(scene["positions"]))
        # Scale to voxel size
        input_coords = input_coords / self.cfg.voxel_size
        # from here on our voxels coordinates represent the center location of the space they discretize
        vox_coords = np.round(input_coords)  # (num_scene_points, 3)
        ret['vox_coords'], vox2point = np.unique(vox_coords, axis=0, return_inverse=True)  # (num_voxels, 3), (num_points)
        # vox2point maps an array organized as vox_coords to an array organized as scene points: num_voxels->num_points

        # NN voxelization to preserve sharp boundaries and sharp voxel to instance association
        # For each 'active' voxel, we find the closest scene point, and associate it.
        # When back-projecting with 'vox2point, points that got dropped in the process, will be overwritten
        # with the associated NN point values of the voxel.
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(input_coords)
        point2vox = nbrs.kneighbors(ret['vox_coords'], return_distance=False)
        point2vox = point2vox.reshape(-1)  # (num_voxels)
        # point2vox maps an array organized as scene points to an array organized as vox_coords: num_points->num_voxels
        # ---------------- Voxelization Code (END) ----------------- #

        #---------------- INPUT FEATURES ----------------- #
        input_feats = [scene["colors"]]

        if self.cfg.use_normals_input:
            input_normals = scene["normals"]
            input_feats.append(input_normals)

        input_feats = np.concatenate(input_feats, 1)
        #### Voxelize the input to the network (scene)
        ret['vox_segments'] = scene['segments'][point2vox]  # (num_voxels)
        ret['vox_features'] = input_feats[point2vox]  # (num_voxels, feature_dim)
        ret['scene'] = scene

        ret['vox_world_coords'] = ret['vox_coords'] * self.cfg.voxel_size + min(0, np.min(scene["positions"]))
        ret['vox2point'] = vox2point
        ret['point2vox'] = point2vox

        unique_vox_segments = None # initialization
        if not self.cfg.do_segment_pooling:
            # compute position of each voxel in original world space of the scene
            ret['input_location'] = ret['vox_world_coords']
            ret['pred2point'] = vox2point
            if 'warning_no_segment_pooling' not in globals().keys():
                globals().update({'warning_no_segment_pooling': True}) # Show warning just once.
                print('WARNING: not tested - dataloader, no segment pooling')
        else:
            # compute mean position of each segment in original world space of the scene

            # define GT per segment
            unique_vox_segments, seg2vox = np.unique(ret['vox_segments'], return_inverse = True) # (unique_vox_segments), (num_voxels)
            seg2point = seg2vox[vox2point]

            segment_middle = np.zeros((unique_vox_segments.shape[0], 3))
            segment_middle.fill(np.nan)
            for i, segment in enumerate(unique_vox_segments):
                segment_coords = ret['vox_world_coords'][segment == ret['vox_segments']]
                segment_middle[i] = np.mean(segment_coords, axis=0)
            assert ~ np.any(np.isnan(segment_middle))

            ret['input_location'] = segment_middle
            ret['seg2point'] = seg2point
            ret['seg2vox'] = seg2vox
            ret['pred2point'] = seg2point

        if self.mode == 'test':
            return ret
        ret['labels'] = labels

        if self.cfg.bb_supervision:
            if unique_vox_segments is None and not self.cfg.point_association:
                unique_vox_segments = np.unique(ret['vox_segments'])  # (unique_vox_segments), (num_voxels)
            self.bbs_supervision(ret, labels, scene, point2vox, unique_vox_segments)
        else:
            self.mask_supervision(ret, labels, point2vox, unique_vox_segments)

        return ret

    def mask_supervision(self, ret, labels, point2vox, unique_vox_segments):
        ret['vox_instances'] = labels['seg2inst'][ret['vox_segments']]
        if not self.cfg.do_segment_pooling:
            ret['gt_semantics'] = labels['semantics'][point2vox]  # (num_voxels)
            ret['gt_bb_bounds'] = labels['bb_bounds'][point2vox]  # (num_voxels, 3)
            gt_bb_centers = labels['bb_centers'][point2vox]  # (num_voxels, 3)
            # compute per voxel center offset in world coordinates

            ret['instance_ids'] = ret['vox_instances']
        else:
            segments_instances = labels['seg2inst'][unique_vox_segments]
            ret['gt_bb_bounds'] = labels['per_instance_bb_bounds'][segments_instances]
            ret['gt_semantics'] = labels['per_instance_semantics'][segments_instances]
            gt_bb_centers = labels['per_instance_bb_centers'][segments_instances]

            ret['instance_ids'] = segments_instances

        ret['gt_bb_offsets'] = gt_bb_centers - ret['input_location']


        # here we use all instances (including invalid ones) that don't have gt unlabeled=0/wall=1/floor=2/ceiling=22/
        # semantic annotation
        ret['fg_instances'] = np.logical_and((ret['gt_semantics'] > 2), (ret['gt_semantics'] != 22)) # (num_voxels)


    def bbs_supervision(self, ret, labels, scene, point2vox, unique_segs):
        inst_per_point, inst_per_seg = self.approx_association(labels, scene, self.cfg.point_association,
                                                               self.cfg.majority_vote, unique_segs, ret)
        ret['pseudo_inst'] = inst_per_point, inst_per_seg # just for visualization / analysis purposes
        if not self.cfg.do_segment_pooling:
            instances = inst_per_point[point2vox] # voxelize
            gt_full_sem = labels['semantics'][point2vox] # full supervision semantics
        else:
            if inst_per_seg is None:
                raise # point_association and seg_pooling is incompatible
            instances = inst_per_seg
            segments_instances = labels['seg2inst'][unique_segs]
            gt_full_sem = labels['per_instance_semantics'][segments_instances] # full supervision semantics

        gt_unlabeled = gt_full_sem == 0 # scannet has missing annotations, don't supervise on those
        fg_instances = instances > -1 # exclude bg and unknown
        ret['fg_instances'] = fg_instances

        # ------------------ GT INSTANCES
        gt_bb_bounds = np.zeros((len(fg_instances), 3))
        gt_bb_bounds[fg_instances] = labels['per_instance_bb_bounds'][instances[fg_instances]]
        ret['gt_bb_bounds'] = gt_bb_bounds

        gt_bb_centers = np.zeros((len(fg_instances), 3))
        gt_bb_centers[fg_instances] = labels['per_instance_bb_centers'][instances[fg_instances]]
        ret['gt_bb_offsets'] = gt_bb_centers - (ret['input_location'] * fg_instances[:,None] + 0)

        # ------------------ GT SEMANTICS
        # zero label is corresponds to 'unlabeled' and is ignored in loss computation
        gt_semantics = np.zeros(len(fg_instances), dtype=np.int)
        # we use semantics of instances only where we have instances
        gt_semantics[fg_instances] = labels['per_instance_semantics'][instances[fg_instances]]
        # for the "-1" (pseudo) background class, we predict, "2" (floor label in original ScanNet)
        gt_semantics[instances == -1] = 2
        # gt_semantics[instances == -2] = 0 this is implicit by having a starting value of 0
        gt_semantics[gt_unlabeled] = 0 # scannet has missing annotations, keep those as 'unlabeled' class

        ret['gt_semantics'] = gt_semantics


    def approx_association(self,labels, scene, point_association, majority_vote, unique_segs, ret):
        # ---------------- FIND APPROX. BOXES to POINT ASSOCIATIONS --------------------------
        # for each point we find its approximate instance id

        # remove bounding boxes from walls / floor / ceiling
        semantics = labels['per_instance_semantics']
        scene_fg = (semantics > 2) & (semantics != 22)  # also excludes unlabeled

        if self.cfg.dropout_boxes:
            # get dropout probabilities
            # data preparation even works in case no instances are kept
            rng = np.random.default_rng(seed=abs(int(scene['name'], 36))) # always drop out the same instances per scene
            dropout_mask = rng.binomial(1, 1 - self.cfg.dropout_boxes, scene_fg.sum()) != 0
            # drop out some foreground instances
            scene_fg[scene_fg] = dropout_mask

        # get bounding boxes
        centers = labels['per_instance_bb_centers'][scene_fg]
        
        bounds = labels['per_instance_bb_bounds'][scene_fg] + 0.005
        min_corner = centers - bounds
        max_corner = centers + bounds
        instance_ids = labels['unique_instances'][scene_fg]  # id of each bounding box

        if self.cfg.noisy_boxes:
            rng = np.random.default_rng(seed=abs(int(scene['name'], 36))) # always use same noise on scene
            # by 2-sigma rule ~95% samples will be in (-noisy_boxes, noisy_boxes), with std_dev = noisy_boxes/2
            min_corner += rng.normal(loc=0, scale=self.cfg.noisy_boxes/2, size= min_corner.shape) # scale is std dev
            max_corner += rng.normal(loc=0, scale=self.cfg.noisy_boxes/2, size= max_corner.shape) # scale is std dev
            ret['noisy_bbs'] = min_corner, max_corner
        # compute per bb, what points are within it
        # put into matrix: bbs x points
        bb_occupancy = is_within_bb(scene['positions'], min_corner[:, None], max_corner[:, None])
        # stores for each point: list of what BBs indices it is contained in
        activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(scene['positions']))]
        # number of BBs a point is in
        num_BBs_per_point = bb_occupancy.sum(axis=0)
        bb_volume = np.prod(2 * bounds, axis=1)
        # WE USE INSTANCE ID -1 for background, -2 for unknown, 0 for GT unlabeled
        if point_association or majority_vote:
            # get bb ID for each point, if point has single BB
            inst_per_point = np.ones(len(scene['positions']), dtype=np.int) * -1
            for i, activ in enumerate(activations_per_point):
                if num_BBs_per_point[i] == 1:  # if point is active in one row (bb)
                    bb_idx = activ[0, 0]
                    inst_per_point[i] = instance_ids[bb_idx]  # add the row idx (bb idx)
                if num_BBs_per_point[i] > 1:  # multiple BBs associated, label unknown
                    if not self.cfg.smallest_bb_heuristic:
                        inst_per_point[i] = -2 # undecided
                    else:
                        box_ids = activ.reshape(-1)
                        smallest_box_id = np.argmin(bb_volume[box_ids])
                        inst_id = instance_ids[box_ids[smallest_box_id]]
                        inst_per_point[i] = inst_id # smallest bb

            if point_association:
                return inst_per_point, None # return per point and per segment annotation

            # WARNING: 'unique_segs' are unique voxel segments, there can be missing segments on point level
            # therefore 'inst_per_point_maj_pooled' can have some incorrectly kept default values
            inst_per_point_maj_pooled = np.ones(len(scene['positions']), dtype=np.int) * -2
            inst_per_seg_maj_pooled = np.ones(len(unique_segs), dtype=np.int) * -2
            for i, seg_id in enumerate(unique_segs):
                seg_mask = seg_id == scene['segments']
                ins_id = stats.mode(inst_per_point[seg_mask], None)[0][0]
                inst_per_point_maj_pooled[seg_mask] = ins_id
                inst_per_seg_maj_pooled[i] = ins_id

            return inst_per_point_maj_pooled, inst_per_seg_maj_pooled
        else:
            # use segments for inference
            # get bb ID for each point, if point lies on a segment which has a point within only a single BB


            # "if any point of the segment is in no BB, then the whole segment is in no BB"
            # WARNING: 'unique_segs' are unique voxel segments, there can be missing segments on point level
            # therefore 'inst_per_point_pooled' can have some incorrectly kept default values
            inst_per_point_pooled = np.ones(len(scene['positions']), dtype=np.int) * -2
            inst_per_seg_pooled = np.ones(len(unique_segs), dtype=np.int) * -2
            for i, seg_id in enumerate(unique_segs):
                seg_mask = seg_id == scene['segments']
                num_BBs_on_seg = num_BBs_per_point[seg_mask]
                min_BBs_on_seg = num_BBs_on_seg.min()
                if min_BBs_on_seg == 1:
                    seg_idx = np.where(num_BBs_on_seg == 1)[0][0]  # get index of point within segment
                    seg_idx2point_idx = np.where(seg_mask)[0]
                    point_idx = seg_idx2point_idx[seg_idx]  # get index of point within scene
                    bb_idx = activations_per_point[point_idx][0, 0]  # get BB of that point

                    inst_per_point_pooled[seg_mask] = instance_ids[bb_idx]  # set the same BB for all seg points
                    inst_per_seg_pooled[i] = instance_ids[bb_idx]  # set the same BB for all seg points
                if min_BBs_on_seg == 0: # no BB, this is background. We set the -1
                    inst_per_point_pooled[seg_mask] = -1
                    inst_per_seg_pooled[i] = -1
                # if min_BBs_on_seg > 1: # multiple BBs associated, we keep the label 'unknown' = -2
                # points that belong to removed segments due to voxelization, will also get the unknown -2 label

            if self.cfg.smallest_bb_heuristic:
                undecided_segs = np.where(inst_per_seg_pooled == -2)[0]
                for undecided_seg in undecided_segs:
                    seg_id = unique_segs[undecided_seg]
                    seg_mask = seg_id == scene['segments']
                    num_BBs_on_seg = num_BBs_per_point[seg_mask]
                    seg_point = num_BBs_on_seg.argmin()
                    point_idx = np.where(seg_mask)[0][seg_point]
                    box_ids = activations_per_point[point_idx].reshape(-1)
                    smallest_box_id = np.argmin(bb_volume[box_ids])
                    inst_id = instance_ids[box_ids[smallest_box_id]]
                    inst_per_seg_pooled[undecided_seg] = inst_id
                    inst_per_point_pooled[seg_mask] = inst_id

            return inst_per_point_pooled, inst_per_seg_pooled # return per point and per segment annotation

    def get_loader(self, shuffle=True, drop_last=True, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.batch_size

        return torch.utils.data.DataLoader(
                self, batch_size=batch_size, num_workers=self.cfg.num_workers, shuffle=shuffle, drop_last=drop_last,
                worker_init_fn=self.worker_init_fn, collate_fn=collate_fn(self.cfg, self.mode))

    def worker_init_fn(self, worker_id):
        if not self.cfg.fixed_seed:
            random_data = os.urandom(4)
            base_seed = int.from_bytes(random_data, byteorder="big")

            np.random.seed(base_seed + worker_id)
            random.seed(base_seed + worker_id)
            torch.random.manual_seed(base_seed + worker_id)
            torch.cuda.manual_seed(base_seed + worker_id)

    def get_loader_multi_gpu(self, rank, world_size, shuffle=True, drop_last=True, batch_size=None):
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(self, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
        if batch_size is None:
            batch_size = self.cfg.batch_size
        return  torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=self.cfg.num_workers, drop_last=drop_last,
            worker_init_fn=self.worker_init_fn, collate_fn=collate_fn(self.cfg, self.mode),sampler=sampler)

class ARKitScenes(Dataset):
    def __init__(self, mode, cfg, do_augmentations=True, subsample_rate=10):
        self.do_augmentations = do_augmentations
        self.mode = mode
        self.cfg = cfg
        self.subsample_rate = subsample_rate

        if mode == 'train':
            files = [f for f in os.listdir(os.path.join(cfg.data_dir, '3dod/Training')) if f.startswith('4')]
            self.data_list = np.array(files)
        elif mode == 'val':
            files = [f for f in os.listdir(os.path.join(cfg.data_dir, '3dod/Validation')) if f.startswith('4')]
                # Some of the example scenes
                # files = ['41069050', #(bath)
                # '41125718',  #(bedroom)
                # '41125763',  # (bath)
                # '41159503',  # (bath, fancy bathtube)
                # '41159566',  # (living room)
                # '41254386',  # (bath)
                # '42445429',  # (living room)
                # '42897542',  # (living room)
                # '42898854']  # (living room)
            self.data_list = np.array(files)
        elif mode == 'predict_specific_scene':
            self.data_list = [cfg.predict_specific_scene,]
        else:
            print('unknown mode')
            exit()

        # for testing purposes: over-fit a model to a single scene, via an integer index
        if cfg.overfit_to_single_scene is not None:
            self.data_list = [self.data_list[cfg.overfit_to_single_scene]] * 100

        # for testing purposes: over-fit a model to a single scene, via a scene name string
        if cfg.overfit_to_single_scene_str is not None:
            self.data_list = [cfg.overfit_to_single_scene_str,] * 100
        if cfg.dataset_size is not None:
            self.data_list = self.data_list[:cfg.dataset_size]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # will store all return data of this method
        ret = {}
        scene_name = self.data_list[idx]

        scene, labels = arkitscenes.process_scene(scene_name, self.mode, self.cfg,
                                                  do_augmentations=self.do_augmentations,
                                                  subsample_rate=self.subsample_rate)

        #---------------- Voxelization Code (START) ----------------- #
        # Translate scene to avoid negative coords
        input_coords = scene["positions"] - min(0, np.min(scene["positions"]))
        # Scale to voxel size
        input_coords = input_coords / self.cfg.voxel_size
        # from here on our voxels coordinates represent the center location of the space they discretize
        vox_coords = np.round(input_coords)  # (num_scene_points, 3)
        ret['vox_coords'], vox2point = np.unique(vox_coords, axis=0, return_inverse=True)  # (num_voxels, 3), (num_points)
        # vox2point maps an array organized as vox_coords to an array organized as scene points: num_voxels->num_points

        # NN voxelization to preserve sharp boundaries and sharp voxel to instance association
        # For each 'active' voxel, we find the closest scene point, and associate it.
        # When back-projecting with 'vox2point, points that got dropped in the process, will be overwritten
        # with the associated NN point values of the voxel.
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(input_coords)
        point2vox = nbrs.kneighbors(ret['vox_coords'], return_distance=False)
        point2vox = point2vox.reshape(-1)  # (num_voxels)
        # point2vox maps an array organized as scene points to an array organized as vox_coords: num_points->num_voxels
        # ---------------- Voxelization Code (END) ----------------- #

        #---------------- INPUT FEATURES ----------------- #
        input_feats = [scene["colors"]]

        if self.cfg.use_normals_input:
            input_normals = scene["normals"]
            input_feats.append(input_normals)

        input_feats = np.concatenate(input_feats, 1)
        #### Voxelize the input to the network (scene)
        ret['vox_segments'] = scene['segments'][point2vox]  # (num_voxels)
        ret['vox_features'] = input_feats[point2vox]  # (num_voxels, feature_dim)
        ret['scene'] = scene

        ret['vox_world_coords'] = ret['vox_coords'] * self.cfg.voxel_size + min(0, np.min(scene["positions"]))
        ret['vox2point'] = vox2point
        ret['point2vox'] = point2vox

        unique_vox_segments = None # initialization
        if not self.cfg.do_segment_pooling:
            # compute position of each voxel in original world space of the scene
            ret['input_location'] = ret['vox_world_coords']
            ret['pred2point'] = vox2point
            if 'warning_no_segment_pooling' not in globals().keys():
                globals().update({'warning_no_segment_pooling': True}) # Show warning just once.
                print('WARNING: not tested - dataloader, no segment pooling')
        else:
            # compute mean position of each segment in original world space of the scene

            # define GT per segment
            unique_vox_segments, seg2vox = np.unique(ret['vox_segments'], return_inverse = True) # (unique_vox_segments), (num_voxels)
            seg2point = seg2vox[vox2point]

            segment_middle = np.zeros((unique_vox_segments.shape[0], 3))
            segment_middle.fill(np.nan)
            for i, segment in enumerate(unique_vox_segments):
                segment_coords = ret['vox_world_coords'][segment == ret['vox_segments']]
                segment_middle[i] = np.mean(segment_coords, axis=0)
            assert ~ np.any(np.isnan(segment_middle))

            ret['input_location'] = segment_middle
            ret['seg2point'] = seg2point
            ret['seg2vox'] = seg2vox
            ret['pred2point'] = seg2point

        if self.mode == 'test':
            return ret
        ret['labels'] = labels

        if self.cfg.bb_supervision:
            if unique_vox_segments is None and not self.cfg.point_association:
                unique_vox_segments = np.unique(ret['vox_segments'])  # (unique_vox_segments), (num_voxels)
            self.bbs_supervision(ret, labels, scene, point2vox, unique_vox_segments)
        else:
            self.mask_supervision(ret, labels, point2vox, unique_vox_segments)

        return ret

    def mask_supervision(self, ret, labels, point2vox, unique_vox_segments):
        ret['vox_instances'] = labels['seg2inst'][ret['vox_segments']]
        if not self.cfg.do_segment_pooling:
            ret['gt_semantics'] = labels['semantics'][point2vox]  # (num_voxels)
            ret['gt_bb_bounds'] = labels['bb_bounds'][point2vox]  # (num_voxels, 3)
            gt_bb_centers = labels['bb_centers'][point2vox]  # (num_voxels, 3)
            # compute per voxel center offset in world coordinates

            ret['instance_ids'] = ret['vox_instances']
        else:
            segments_instances = labels['seg2inst'][unique_vox_segments]
            ret['gt_bb_bounds'] = labels['per_instance_bb_bounds'][segments_instances]
            ret['gt_semantics'] = labels['per_instance_semantics'][segments_instances]
            gt_bb_centers = labels['per_instance_bb_centers'][segments_instances]

            ret['instance_ids'] = segments_instances

        ret['gt_bb_offsets'] = gt_bb_centers - ret['input_location']


        # here we use all instances (including invalid ones) that don't have gt unlabeled=0/wall=1/floor=2/ceiling=22/
        # semantic annotation
        # ret['fg_instances'] = np.logical_and((ret['gt_semantics'] > 2), (ret['gt_semantics'] != 22)) # (num_voxels)
        ret['fg_instances'] = (ret['gt_semantics'] > 2)  # (num_voxels)

    def bbs_supervision(self, ret, labels, scene, point2vox, unique_segs):
        """ Bounding box supervision. Generates the per-point / per-segment instance labels.
        ret - not sure
        labels - the ground truth labels from the dataloader (here only GT bounding boxes)
        scene - the input scene from the dataloader
        point2vox - mapping from points to voxels 
        unique_segs - unique segment ids
        """
        inst_per_point, inst_per_seg = self.approx_association(labels, scene, self.cfg.point_association, unique_segs)
        if not self.cfg.do_segment_pooling:
            instances = inst_per_point[point2vox]  # voxelize
            gt_full_sem = labels['semantics'][point2vox]  # full supervision semantics
        else:
            if inst_per_seg is None:
                raise  # point_association and seg_pooling is incompatible
            instances = inst_per_seg

        #gt_unlabeled = gt_full_sem == 0  # scannet has missing annotations, don't supervise on those
        fg_instances = instances > -1  # exclude bg and unknown
        ret['fg_instances'] = fg_instances

        # ------------------ GT INSTANCES
        gt_bb_bounds = np.zeros((len(fg_instances), 3))
        gt_bb_bounds[fg_instances] = labels['per_instance_bb_bounds'][instances[fg_instances]]
        ret['gt_bb_bounds'] = gt_bb_bounds

        gt_bb_centers = np.zeros((len(fg_instances), 3))
        gt_bb_centers[fg_instances] = labels['per_instance_bb_centers'][instances[fg_instances]]
        ret['gt_bb_offsets'] = gt_bb_centers - (ret['input_location'] * fg_instances[:, None] + 0)

        # ------------------ GT SEMANTICS
        # zero label is corresponds to 'unlabeled' and is ignored in loss computation
        gt_semantics = np.zeros(len(fg_instances), dtype=int)
        # we use semantics of instances only where we have instances
        gt_semantics[fg_instances] = labels['per_instance_semantics'][instances[fg_instances]]
        # for the "-1" (pseudo) background class, we predict, "2" (floor label in original ScanNet)
        gt_semantics[instances == -1] = 2
        # gt_semantics[instances == -2] = 0 this is implicit by having a starting value of 0
        # gt_semantics[gt_unlabeled] = 0  # scannet has missing annotations, keep those as 'unlabeled' class

        ret['gt_semantics'] = gt_semantics

    def approx_association(self, labels, scene, point_association, unique_segs):
        # ---------------- FIND APPROX. BOXES to POINT ASSOCIATIONS --------------------------
        # for each point we find its approximate instance id

        # get bounding boxes
        instance_ids = labels['unique_instances']  # id of each bounding box
        centers = labels['per_instance_bb_centers']   # masking not needed here
        bounds = labels['per_instance_bb_bounds'] + 0.05
        rotations = labels['per_instance_bb_rotations']

        inst_per_point_pooled = np.ones(len(scene['positions']), dtype=np.int) * -1
        inst_per_seg_pooled = np.ones(len(unique_segs), dtype=np.int) * -2

        bb_occupancy = np.zeros([rotations.shape[0], scene['positions'].shape[0]], dtype=bool)
        for i in range(rotations.shape[0]):
            pc = scene['positions'] - centers[i]
            rot = np.reshape(rotations[i], [3, 3])
            bb_occupancy[i] = is_within_bb((rot @ pc.T).T, -bounds[i, :], bounds[i, :])

        # compute per bb, what points are within it
        # put into matrix: bbs x points

        # bb_occupancy = is_within_bb(scene['positions'], min_corner[:, None], max_corner[:, None])
        # stores for each point: list of what BBs indices it is contained in
        activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(scene['positions']))]
        # number of BBs a point is in
        num_BBs_per_point = bb_occupancy.sum(axis=0)

        # WE USE INSTANCE ID -1 for background, -2 for unknown, 0 for GT unlabeled
        majority_voting = False
        if point_association:
            # get bb ID for each point, if point has single BB
            inst_per_point = np.ones(len(scene['positions']), dtype=np.int) * -1
            for i, activ in enumerate(activations_per_point):
                if num_BBs_per_point[i] == 1:  # if point is active in one row (bb)
                    bb_idx = activ[0, 0]
                    inst_per_point[i] = instance_ids[bb_idx]  # add the row idx (bb idx)
                if num_BBs_per_point[i] > 1:  # multiple BBs associated, label unknown
                    inst_per_point[i] = -2
                # if min_BBs_on_seg == 0: # no BB, this is background. We keep the -1 we initialized with
            return inst_per_point, None  # return per point and per segment annotation
        elif majority_voting:
            # get bb ID for each point, if point has single BB
            inst_per_point = np.ones(len(scene['positions']), dtype=np.int) * -1
            for i, activ in enumerate(activations_per_point):
                if num_BBs_per_point[i] == 1:  # if point is active in one row (bb)
                    bb_idx = activ[0, 0]
                    inst_per_point[i] = instance_ids[bb_idx]  # add the row idx (bb idx)
                if num_BBs_per_point[i] > 1:  # multiple BBs associated, label unknown
                    inst_per_point[i] = -2
                # if min_BBs_on_seg == 0: # no BB, this is background. We keep the -1 we initialized with
                for i, seg_id in enumerate(unique_segs):
                    seg_mask = seg_id == scene['segments']
                    ins_id = stats.mode(inst_per_point[seg_mask], None)[0][0]
                    inst_per_point_pooled[seg_mask] = ins_id
                    inst_per_seg_pooled[i] = ins_id
            return inst_per_point_pooled, inst_per_seg_pooled  # return per point and per segment annotation
        else:
            # use segments for inference
            # get bb ID for each point, if point lies on a segment which has a point within only a single BB

            # "if any point of the segment is in no BB, then the whole segment is in no BB"
            inst_per_point_pooled = np.ones(len(scene['positions']), dtype=int) * -2
            inst_per_seg_pooled = np.ones(len(unique_segs), dtype=int) * -2
            for i, seg_id in enumerate(unique_segs):
                seg_mask = seg_id == scene['segments']
                num_BBs_on_seg = num_BBs_per_point[seg_mask]
                min_BBs_on_seg = num_BBs_on_seg.min()
                if min_BBs_on_seg == 1:
                    seg_idx = np.where(num_BBs_on_seg == 1)[0][0]  # get index of point within segment
                    seg_idx2point_idx = np.where(seg_mask)[0]
                    point_idx = seg_idx2point_idx[seg_idx]  # get index of point within scene
                    bb_idx = activations_per_point[point_idx][0, 0]  # get BB of that point

                    inst_per_point_pooled[seg_mask] = instance_ids[bb_idx]  # set the same BB for all seg points
                    inst_per_seg_pooled[i] = instance_ids[bb_idx]  # set the same BB for all seg points
                if min_BBs_on_seg == 0:  # no BB, this is background. We set the -1
                    inst_per_point_pooled[seg_mask] = -1
                    inst_per_seg_pooled[i] = -1
                # if min_BBs_on_seg > 1: # multiple BBs associated, we keep the label 'unknown' = -2
                # points that belong to removed segments due to voxelization, will also get the unknown -2 label

            return inst_per_point_pooled, inst_per_seg_pooled # return per point and per segment annotation

    def get_loader(self, shuffle=True, drop_last=True, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.batch_size

        return torch.utils.data.DataLoader(
                self, batch_size=batch_size, num_workers=self.cfg.num_workers, shuffle=shuffle, drop_last=drop_last,
                worker_init_fn=self.worker_init_fn, collate_fn=collate_fn(self.cfg, self.mode))

    def worker_init_fn(self, worker_id):
        if not self.cfg.fixed_seed:
            random_data = os.urandom(4)
            base_seed = int.from_bytes(random_data, byteorder="big")

            np.random.seed(base_seed + worker_id)
            random.seed(base_seed + worker_id)
            torch.random.manual_seed(base_seed + worker_id)
            torch.cuda.manual_seed(base_seed + worker_id)

class collate_fn:
    """Generates collate function for coords, feats, labels.
    """

    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode

    def __call__(self, dict_data):
        """ Collate input coords and input colors into batched format, sampled points are reshaped as a linear array (remove batch axis).
            All outputs are torch tensors. If the number of points in a batch is greater than limit_numpoints, drop some of the samples in the batch
        """
        ret = {}
        for batch_id, _ in enumerate(dict_data):
            for key in dict_data [batch_id].keys ():
                if key not in ret:
                    ret[key] = []
                ret [key].append (dict_data [batch_id][key])

        # Convert coords: (x, y, z) to batched_coords: (batch_id, x, y, z)
        ret["vox_coords"] = ME.utils.batched_coordinates(ret["vox_coords"], dtype=torch.int32)
        ret["vox_features"] = torch.from_numpy(np.concatenate(ret["vox_features"], 0)).float()

        if self.cfg.do_segment_pooling:
            segment_batch_ids = []
            for batch_id, input_location in enumerate(ret['input_location']):
                segment_batch_ids.append([batch_id] * len(input_location))  # maps segment to batch
            # used to decompose the concatenated results into seperate scenes
            ret['batch_ids'] = torch.from_numpy(np.concatenate(segment_batch_ids, 0)).long()
            #     # used to compute GT masks in segment space for eval (in training code) and gt score (in prediction code)
            #     segments_instances = torch.from_numpy(np.concatenate(segments_instances, 0)).long()  # (num_voxels_in_batch)
        else:
            ret['batch_ids'] = ret["vox_coords"][:, 0]

        ret['input_location'] = torch.from_numpy(np.concatenate(ret['input_location'], 0)).float()
        ret['pooling_ids'] = to_unique(ret['vox_segments'])


        if self.mode == 'test':
            return ret

        # GT
        ret["gt_bb_bounds"] = torch.from_numpy(np.concatenate(ret["gt_bb_bounds"], 0)).float()
        ret["gt_bb_offsets"] = torch.from_numpy(np.concatenate(ret["gt_bb_offsets"], 0)).float()
        ret['gt_semantics'] = torch.from_numpy(np.concatenate(ret['gt_semantics'], 0)).long()
        ret['fg_instances'] = torch.from_numpy(np.concatenate(ret['fg_instances'], 0)).bool()
        return ret