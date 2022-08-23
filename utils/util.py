import numpy as np
import torch
import copy

def get_bb_lines(bb_center, bb_bounds):
    start_list = []
    end_list = []
    bb_min = bb_center - bb_bounds
    bb_max = bb_center + bb_bounds
    length = bb_max - bb_min
    for i in range(3):
        start_list.append(bb_min)
        end = bb_min.copy()
        end[i] = bb_max[i]
        end_list.append(end)
        for j in range(3):
            if i == j:
                continue
            start_list.append(end)
            second_end = end.copy()
            second_end[j] += length[j]
            end_list.append(second_end)
    for i in range(3):
        start_list.append(bb_max)
        end = bb_max.copy()
        end[i] = bb_min[i]
        end_list.append(end)
    return np.array(start_list), np.array(end_list)

def get_bbs_lines(bbs_centers, bbs_bounds):
    if type(bbs_centers) is torch.Tensor or  type(bbs_bounds) is torch.Tensor:
        bbs_centers = bbs_centers.numpy()
        bbs_bounds = bbs_bounds.numpy()
    start_list = []
    end_list = []
    for i in range(len(bbs_centers)):
        # for i in range(len(bb_centers)):
        start, end = get_bb_lines(bbs_centers[i], bbs_bounds[i])
        start_list.append(start)
        end_list.append(end)
    start = np.concatenate(start_list, 0)
    end = np.concatenate(end_list, 0)
    return start, end

# out: bb [min corner ,max corner, score]
def to_bbs_min_max(locations, offsets, bounds, scores=None, use_torch=True):
    if use_torch:
        centers = offsets + locations
        if offsets.is_cuda:
            bbs = torch.cuda.FloatTensor(centers.shape[0], 6).fill_(0)
        else:
            bbs = torch.zeros((centers.shape[0], 6))
        bbs[:, :3] = centers - bounds
        bbs[:, 3:] = centers + bounds
        if scores is not None:
            bbs = torch.cat((scores, bbs), axis=1)
    else:
        centers = offsets + locations
        bbs = np.zeros((centers.shape[0], 6))
        bbs[:, :3] = centers - bounds
        bbs[:, 3:] = centers + bounds
        if scores is not None:
            bbs = np.concatenate((scores, bbs), axis=1)
    return bbs

def to_bbs_min_max_(centers, bounds, device):
    bounding_boxes = torch.zeros((bounds.shape[0], 6), device=device)
    bounding_boxes[:, :3] = centers - bounds
    bounding_boxes[:, 3:] = centers + bounds
    return bounding_boxes

# go from min_corner, max_corner representation to center, bounds representation
def to_bb_center_a_bounds(bbs_min_max):
    bb_centers = (bbs_min_max[:,3:] + bbs_min_max[:,:3]) / 2
    bb_bounds = bbs_min_max[:,3:] - bb_centers
    return bb_centers, bb_bounds

def get_all_bb_corners(bb_centers,bb_bounds):
    # powerset of all dimensions
    neg_dims =  [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    corner_displacements = bb_bounds.expand(len(neg_dims),-1,-1) # (8,num_predictions on all scenes,3)
    for i, neg_dim in enumerate(neg_dims):
        corner_displacements[i,:,neg_dim] *= -1
    eight_cornered_bbs = bb_centers + corner_displacements # (8,num_predictions on all scenes,3)
    return eight_cornered_bbs

# works on torch tensors
def is_within_bb(points, bb_min, bb_max):
    return torch.all( points >= bb_min, axis=-1) & torch.all( points <= bb_max, axis=-1)
# numpy version
def is_within_bb_np(points, bb_min, bb_max):
    return np.all( points >= bb_min, axis=-1) & np.all( points <= bb_max, axis=-1)

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds

import random
from collections import defaultdict

colors = defaultdict(lambda: [random.random() * 255, random.random() * 255, random.random() * 255])
colors[0] = [0,0,0]
colors[-2] = [255,0,0]
def to_color(arr):
    return np.array([colors[e] for e in arr])

def scalar2colors(arr):
    colors = np.zeros((len(arr),3))
    colors[:,1] = arr
    colors *= 255
    return colors

def to_worldcoords(vox_coords,scene, cfg):
    return (vox_coords * cfg.voxel_size + min(0, np.min(scene["positions"]))).numpy()

# ----------------- map segment ids to dense ranking starting at index 0 (needed by ME global pool function)
# Segment ids can be duplicates: map segment ids to unique ones.
# This means, that every segment in each batch, needs to have a unique batch_id, in order to be pooled
# separately.

def to_unique( segments): # enumeration_ids, when we have id arrays, like [0,1,2,..,n]
    unique_segments =  copy.deepcopy(segments)
    # make sure all segments across scenes have unique ids
    for i in range(1, len(unique_segments)):
        unique_segments[i] += np.max(unique_segments[i - 1]) + 1
    unique_segments = np.concatenate(unique_segments, 0)
    _, pooling_ids = np.unique(unique_segments, return_inverse=True)
    return torch.from_numpy(pooling_ids).long()


# Epoch counts from 0 to N-1
from math import cos, pi
def cosine_lr_after_step(optimizer, base_lr, epoch, start_epoch, total_epochs, clip=1e-6):
    if epoch < start_epoch:
        lr = base_lr
    else:
        lr =  clip + 0.5 * (base_lr - clip) * \
            (1 + cos(pi * ( (epoch - start_epoch) / (total_epochs - start_epoch))))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr