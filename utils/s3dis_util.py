import sys
sys.path.append('.')

import os
import numpy as np
from scipy import stats
from sklearn.cluster import MeanShift, DBSCAN  
import pyviz3d.visualizer as viz 
import dataprocessing.scannet as scannet # Using scannet color mapping
import dataprocessing.s3dis as s3dis # Using scannet color mapping
from matplotlib import cm as viz_cm
from sklearn.neighbors import NearestNeighbors
from utils.util import get_bbs_lines

import open3d as o3d

NUM_CLASSES = 13

def reconstruct_mesh (scene):
    positions = scene ["positions"]
    normals = scene ["normals"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype (np.float32))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype (np.float32))
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
    return mesh

def interpolate(original_scene, sampled_positions, sampled_colors_list, radius=0.1, neutral_color=[1.0, 1.0, 1.0]):
    import copy
    pcd_tree = o3d.geometry.KDTreeFlann(original_scene)
    num_querries = sampled_positions.shape[0]
    interpolated_scenes_list = [copy.deepcopy(original_scene) for _ in range(len(sampled_colors_list))]
    for j in range(len(interpolated_scenes_list)):
        np.asarray(interpolated_scenes_list[j].vertex_colors)[:, :] = np.array(neutral_color)
    mesh_pos = np.asarray(original_scene.vertices)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sampled_positions)
    dist, original2sample = nbrs.kneighbors(mesh_pos)
    dist = dist [:, 0]
    original2sample = original2sample[:,0]
    for j in range(len(interpolated_scenes_list)):
        mesh = interpolated_scenes_list [j]
        original2sample [dist < radius]
        interpolated_colors = sampled_colors_list [j][original2sample [dist < radius]]
        colors = np.asarray(mesh.vertex_colors)
        colors [dist < radius] = interpolated_colors
        interpolated_scenes_list[j].vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return interpolated_scenes_list

def visualize_prediction (cfg, scene_name, scene, labels, pred_label, out_path):
    # ---------------- GET GT INSTANCES, SEMANTICS, and BBs
    print ("visualize ", scene_name)

    color_map = viz_cm.get_cmap('Paired', 12)
    colors_map = np.array(color_map(range(12)))[:, :3]
    r, g, b = colors_map.T
    colors_map = np.vstack((colors_map, np.array([r, b, g]).T, np.array([ b, r, g]).T, np.array([ g, r, b]).T, np.array([ b, g, r]).T, np.array([ g, b, r]).T))
    colors_map = np.vstack((colors_map,colors_map,colors_map))
    if np.max (colors_map) < 2:
        colors_map = (colors_map * 255).astype (np.int32)
    INS_COLORS = colors_map

    # Using color map from scannet
    SEM_COLORS = np.copy (scannet.scannet_color_map).astype (np.float32)
    SEM_COLORS [0] = SEM_COLORS [-2]

    scannet.scannet_color_map = s3dis.S3DIS_SEMANTICS_COLORS

    gt_inst_colors = INS_COLORS [labels['instances']] 
    sem = labels['semantics']
    gt_sem_colors = scannet.scannet_color_map[sem]

    gt_inst_colors = gt_inst_colors / 255
    gt_sem_colors = gt_sem_colors / 255

    mesh = reconstruct_mesh (scene)

    instance_fg = s3dis.semantics_to_forground_mask (labels['per_instance_semantics'], cfg)

    bbs = np.hstack((labels['per_instance_bb_centers'][instance_fg], 2* labels['per_instance_bb_bounds'][instance_fg], 
                                    scannet.scannet_color_map[labels['per_instance_semantics'][instance_fg]])).T

    # ---------------- GET INSTANCES AND SEMANTICS COLORS
    pred_inst_colors = np.ones((len(scene ["positions"]), 3)) * 255
    pred_sem_colors = np.ones((len(scene ["positions"]), 3)) * 255

    for ins_id in np.unique (pred_label ["instances"]):
        mask = pred_label ["instances"] == ins_id
        sem_label = stats.mode(pred_label ["semantics"][mask], None)[0][0]
        if ins_id < 1:
            pred_inst_colors [mask] = [255,255,255]
        else:
            gt_ins_id = stats.mode(labels['instances'][mask], None)[0][0]
            pred_inst_colors [mask] = INS_COLORS [gt_ins_id]

    pred_sem_colors = scannet.scannet_color_map[pred_label ["semantics"]]
    pred_sem_colors [pred_label ["semantics"] < 0] = [255, 255, 255]

    pred_sem_colors = pred_sem_colors / 255
    pred_inst_colors = pred_inst_colors / 255

    mesh_rgb, mesh_gt_sem, mesh_gt_ins, mesh_pred_sem, mesh_pred_ins= interpolate (mesh, scene["positions"], 
                            [scene ["colors"], gt_sem_colors, gt_inst_colors, pred_sem_colors, pred_inst_colors], 
                            0.04)
    interp_colors = np.asarray(mesh_gt_sem.vertex_colors)
    void_mask = interp_colors.sum (1) == 3.0
    mesh_rgb.remove_vertices_by_index (np.where (void_mask) [0])
    mesh_gt_sem.remove_vertices_by_index (np.where (void_mask) [0])
    mesh_gt_ins.remove_vertices_by_index (np.where (void_mask) [0])
    mesh_pred_sem.remove_vertices_by_index (np.where (void_mask) [0])
    mesh_pred_ins.remove_vertices_by_index (np.where (void_mask) [0])

    o3d.io.write_triangle_mesh(os.path.join(out_path,'rgb.ply'), mesh_rgb)

    o3d.io.write_triangle_mesh(os.path.join(out_path,'gt_instances.ply'), mesh_gt_ins)
    o3d.io.write_triangle_mesh(os.path.join(out_path,'gt_semantics.ply'), mesh_gt_sem)

    o3d.io.write_triangle_mesh(os.path.join(out_path,'pred_instances.ply'), mesh_pred_ins)
    o3d.io.write_triangle_mesh(os.path.join(out_path,'pred_semantics.ply'), mesh_pred_sem)

    # -------------------- SAVE IN PYVIZ --------------------
    v = viz.Visualizer()
    v.add_points(f'Input scene', scene['positions'], scene['colors'] * 255, point_size=25, visible=False)
    v.add_points(f'GT Instances',scene['positions'], gt_inst_colors * 255, point_size=25, visible=False)
    v.add_points(f'GT Semantics',scene['positions'], gt_sem_colors * 255, point_size=25, visible=False)
    v.add_points(f'Pred Instances',scene['positions'],pred_inst_colors * 255, point_size=25, visible=False)
    v.add_points(f'Pred Semantics',scene['positions'], pred_sem_colors * 255, point_size=25, visible=False)
    start, end = get_bbs_lines(labels['per_instance_bb_centers'][instance_fg], labels['per_instance_bb_bounds'][instance_fg])
    bbs_colors = np.repeat(scannet.scannet_color_map[labels['per_instance_semantics'][instance_fg]], 12, axis=0)
    v.add_lines(f'GT BBs', start, end, bbs_colors, visible=False)
    v.save(os.path.join(out_path,'pyviz3d'), verbose=False)
    print ('Pyviz visualization to ', os.path.join(out_path,'pyviz3d'))

def assign_semantics_to_proposals (pred_semantics, proposal_masks):
    # Use majoring vote to determind the semantic of proposals
    proposal_semantics = []
    for mask in proposal_masks:
        mask = mask > 0
        semantic_id = np.bincount (pred_semantics [mask]).argmax ()
        proposal_semantics.append (semantic_id)
    return np.array (proposal_semantics)

def clustering_for_background (pred_semantics, coords, normals):
    ''' For the S3DIS scene:
        - we use DBSCAN to cluster the instances of walls
        - we use the semantic prediction to get the the floor insance and the ceiling instance (only 1 ceiling and 1 floor in each scene)
        - Non-maximum-clustering / bounding boxes are not used / predicted for walls / floors / ceiling
    '''
    pred_instances = np.zeros_like (pred_semantics).astype (np.int32)
    # instance ID of ceiling
    pred_instances [pred_semantics == 0] = 1
    # instance ID of floor
    pred_instances [pred_semantics == 1] = 2

    wall_mask = pred_semantics == 2
    wall_coords = coords [wall_mask]
    wall_normals = normals [wall_mask] * 2 # priotizing normal over coordinates
    wall_features = np.concatenate ([wall_coords, wall_normals], 1)

    # wall_clustering = MeanShift(bandwidth=2, n_jobs=16).fit(wall_features)
    wall_clustering = DBSCAN(eps=0.35, min_samples=10, n_jobs=16).fit(wall_features)
    wall_clustering.labels_ = wall_clustering.labels_ + 4
    wall_instances = wall_clustering.labels_
    
    # remove small noises
    bg_prop_ids, bg_prop_cnts = np.unique (wall_instances, return_counts=True)
    wall_id_small_mask = bg_prop_cnts < 3000
    small_prop_ids = bg_prop_ids [wall_id_small_mask]
    wall_small_mask = np.isin (wall_instances, small_prop_ids)
    wall_instances [wall_small_mask] = -1

    pred_instances [wall_mask] = wall_instances

    return pred_instances

def s3dis_eval (pred_labels, gt_labels):

    num_room = len(gt_labels)

    # Initialize...
    # acc and macc
    total_true = 0
    total_seen = 0
    true_positive_classes = np.zeros(NUM_CLASSES)
    positive_classes = np.zeros(NUM_CLASSES)
    gt_classes = np.zeros(NUM_CLASSES)
    # mIoU
    ious = np.zeros(NUM_CLASSES)
    totalnums = np.zeros(NUM_CLASSES)
    # precision & recall
    total_gt_ins = np.zeros(NUM_CLASSES)
    at = 0.5
    tpsins = [[] for itmp in range(NUM_CLASSES)]
    fpsins = [[] for itmp in range(NUM_CLASSES)]
    # mucov and mwcov
    all_mean_cov = [[] for itmp in range(NUM_CLASSES)]
    all_mean_weighted_cov = [[] for itmp in range(NUM_CLASSES)]


    for i in range(num_room):
        data_label = pred_labels [i]
        pred_ins = pred_labels [i]["instances"]
        pred_sem = pred_labels [i]["semantics"]
        gt_label = gt_labels [i]
        gt_ins = gt_label ["instances"]
        gt_sem = gt_label ["semantics"]

        # semantic acc
        total_true += np.sum(pred_sem == gt_sem)
        total_seen += pred_sem.shape[0]

        # pn semantic mIoU
        for j in range(gt_sem.shape[0]):
            gt_l = int(gt_sem[j])
            pred_l = int(pred_sem[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l==pred_l)

        # instance
        un = np.unique(pred_ins)
        pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = (pred_ins == g)
            sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]

        un = np.unique(gt_ins)
        pts_in_gt = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):
            tmp = (gt_ins == g)
            sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
            pts_in_gt[sem_seg_i] += [tmp]
        # NOTE: 
        # pts_in_gt: (Nclass, Npoints) - binary array, gt instance list of each gt class
        # pts_in_pred: (Nclass, Npoints) - binary array, pred instance list of each pred class

        # instance mucov & mwcov
        for i_sem in range(NUM_CLASSES):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                ovmax = 0.
                num_ins_gt_point = np.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        ipmax = ip

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                all_mean_cov[i_sem].append(mean_cov)

                mean_weighted_cov /= num_gt_point
                all_mean_weighted_cov[i_sem].append(mean_weighted_cov)


        # instance precision & recall
        for i_sem in range(NUM_CLASSES):
            tp = [0.] * len(pts_in_pred[i_sem])
            fp = [0.] * len(pts_in_pred[i_sem])
            gtflag = np.zeros(len(pts_in_gt[i_sem]))
            total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                ovmax = -1.

                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)


                    if iou > ovmax:
                        ovmax = iou
                        igmax = ig

                if ovmax >= at:
                        tp[ip] = 1  # true
                else:
                    fp[ip] = 1  # false positive

            tpsins[i_sem] += tp
            fpsins[i_sem] += fp


    MUCov = np.zeros(NUM_CLASSES)
    MWCov = np.zeros(NUM_CLASSES)
    for i_sem in range(NUM_CLASSES):
        MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
        MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

    precision = np.zeros(NUM_CLASSES)
    recall = np.zeros(NUM_CLASSES)
    for i_sem in range(NUM_CLASSES):
        tp = np.asarray(tpsins[i_sem]).astype(np.float)
        fp = np.asarray(fpsins[i_sem]).astype(np.float)
        tp = np.sum(tp)
        fp = np.sum(fp)

        rec = tp / total_gt_ins[i_sem]
        prec = tp / (tp + fp)

        precision[i_sem] = prec
        recall[i_sem] = rec

    def log_string(out_str):
        print(out_str)

    log_string('Instance Segmentation Precision: {}'.format(precision))
    log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision)))
    log_string('Instance Segmentation Recall: {}'.format(recall))
    log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall)))



    # semantic results
    iou_list = []
    for i in range(NUM_CLASSES):
        iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
        iou_list.append(iou)

    return np.mean(precision), np.mean(recall), precision, recall