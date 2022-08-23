import torch
import numpy as np

def set_IOUs(boxes_a, boxes):
    # assert boxes are defined as: (min_corner, max_corner)
    assert boxes_a.shape[1] == 6 and boxes.shape[1] == 6
    boxes_a_side_lengths = boxes_a[:, 3:] - boxes_a[:, :3]
    boxes_side_lengths = boxes[:, 3:] - boxes[:, :3]
    assert torch.all(boxes_a_side_lengths >= 0) and torch.all(boxes_side_lengths >= 0)

    intersection_min = torch.maximum(boxes_a[:, :3], boxes[:, :3])
    intersection_max = torch.minimum(boxes_a[:, 3:], boxes[:, 3:])

    # no overlap produces negative values, and is cutoff by 0
    intersection_side_lengths = torch.clamp( intersection_max - intersection_min, min=0)
    intersection_area = torch.prod(intersection_side_lengths, axis=1)

    boxes_a_area = torch.prod(boxes_a_side_lengths, axis=1)
    boxes_area = torch.prod(boxes_side_lengths, axis=1)

    union_area = boxes_a_area + boxes_area - intersection_area + 0.000001
    return intersection_area / union_area


# axis-aligned bounding boxes only
def torch_IOUs(box, boxes):
    # assert boxes are defined as: (min_corner, max_corner)
    assert box.shape[0] == 6 and boxes.shape[1] == 6

    box_side_lengths = box[3:] - box[:3]
    boxes_side_lengths = boxes[:, 3:] - boxes[:, :3]
    assert torch.all(box_side_lengths >= 0) and torch.all(boxes_side_lengths >= 0)

    intersection_min = torch.maximum(box[:3], boxes[:, :3])
    intersection_max = torch.minimum(box[3:], boxes[:, 3:])

    # no overlap produces negative values, and is cutoff by 0
    intersection_side_lengths = torch.clamp( intersection_max - intersection_min, min=0)
    intersection_area = torch.prod(intersection_side_lengths, axis=1)

    box_area = torch.prod(box_side_lengths)
    boxes_area = torch.prod(boxes_side_lengths, axis=1)

    union_area = box_area + boxes_area - intersection_area + 0.000001
    return intersection_area / union_area


def np_NMS_clustering(boxes, cluster_th=0.5):
    # boxes should be a list of 3D boxes [box_score, min_corner,max_corner], higher scores for better boxes
    assert boxes.shape[1] == 7 and len(boxes.shape) == 2
    assert cluster_th > 0 and cluster_th < 1
    remaining_boxes_indices = np.argsort(-boxes[:, 0])
    clusters = []

    while len(remaining_boxes_indices) > 0:
        remaining_boxes = boxes[remaining_boxes_indices]
        # remove score component
        remaining_boxes = remaining_boxes[:, 1:]
        ious = IOUs(remaining_boxes[0], remaining_boxes)
        iou_mask = ious <= cluster_th

        clusters.append([remaining_boxes_indices[0], remaining_boxes_indices[~iou_mask]])
        remaining_boxes_indices = remaining_boxes_indices[iou_mask]

    return clusters


def NMS_clustering(boxes, cluster_th=0.5, get_heatmaps=True):
    # boxes should be a list of 3D boxes [box_score, min_corner,max_corner], higher scores for better boxes
    assert boxes.shape[1] == 7 and len(boxes.shape) == 2
    assert cluster_th > 0 and cluster_th < 1
    # boxes should have positive side lengths - otherwise they don't have an area and are invalid
    boxes_side_lengths = boxes[:, 4:] - boxes[:, 1:4]
    valid = torch.min(boxes_side_lengths, axis=1)[0] > 0 # (num_boxes)
    if ~ torch.all(valid):
        print('Warning: Invalid boxes found.')

    remaining_boxes_indices = torch.argsort(-boxes[:, 0])
    # remove score component
    boxes = boxes[:, 1:]
    cluster_representant = []
    clusters = []
    cluster_heatmaps = []
    while len(remaining_boxes_indices) > 0:
        #print(len(remaining_boxes_indices))
        remaining_boxes = boxes[remaining_boxes_indices]
        if get_heatmaps:
            cluster_heatmap = torch_IOUs(remaining_boxes[0], boxes)
            # manually set iou to 1, even for invalid boxes (side_lengths <=0)
            cluster_heatmap[remaining_boxes_indices[0]] = 1
            cluster_heatmaps.append(cluster_heatmap)
            ious = cluster_heatmap[remaining_boxes_indices]
        else:
            ious = torch_IOUs(remaining_boxes[0], remaining_boxes)
            # manually set iou to 1, even for invalid boxes (side_lengths <=0)
            ious[0] = 1
        iou_mask = ious <= cluster_th
        cluster_representant.append(remaining_boxes_indices[0])
        clusters.append(remaining_boxes_indices[~iou_mask])
        remaining_boxes_indices = remaining_boxes_indices[iou_mask]

    if get_heatmaps:
        return torch.Tensor(cluster_representant).long(), clusters, torch.stack(cluster_heatmaps,0)
    else:
        return torch.Tensor(cluster_representant).long(), clusters


# input masks: bool (true inside, false outside), shape: (num_masks, num_mask_elements)
def masks_iou(mask, masks, allow_empty = False):
    # empty masks are invalid
    if not allow_empty:
        assert torch.all(torch.sum(masks, axis=1) > 0) and torch.sum(mask) > 0
        intersection = torch.sum(mask & masks, axis=1)
        union = torch.sum(mask | masks, axis=1)
        return intersection / union
    else:
        intersection = torch.sum(mask & masks, axis=1)
        union = torch.sum(mask | masks, axis=1)
        ret = torch.zeros_like(union).float()
        ret[union > 0] = intersection[union > 0] / union[union > 0]
        return ret

def mask_iou_np(mask, mask_b):
    # empty masks are invalid
    assert np.sum(mask_b) > 0 and np.sum(mask) > 0
    intersection = np.sum(mask & mask_b)
    union = np.sum(mask | mask_b)
    return intersection / union

def mask_NMS(sorted_masks, cluster_th=0.5, allow_empty = False):
    remaining_masks_indices = torch.arange(len(sorted_masks))
    output_masks = []
    suppressed = []
    while len(remaining_masks_indices) > 0:
        remaining_masks = sorted_masks[remaining_masks_indices]
        ious = masks_iou(remaining_masks[0], remaining_masks, allow_empty)
        ious[0] = 1
        iou_mask = ious <= cluster_th

        output_masks.append(remaining_masks_indices[0])
        suppressed.append((remaining_masks_indices[0], remaining_masks_indices[~iou_mask]))
        remaining_masks_indices = remaining_masks_indices[iou_mask]

    return torch.hstack(output_masks), suppressed

def semIOU(pred_label, gt_label):
    IOU = []
    # ignore invalid and unlabeled regions
    valid = gt_label > -100
    gt_label = gt_label[valid]
    pred_label = pred_label[valid]
    scene_labels = torch.unique(torch.cat((gt_label,pred_label)))
    for l in scene_labels:
        intersection = torch.sum((pred_label == l) & (gt_label == l))
        union = torch.sum((pred_label == l) | (gt_label == l))
        IOU.append((intersection / (union + 1e-6)).item())
    return np.array(IOU)
