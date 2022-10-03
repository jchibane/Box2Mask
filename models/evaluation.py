import sys
sys.path.append('.')

from models.dataloader import ScanNet, ARKitScenes, S3DIS
import torch
import dataprocessing.scannet as scannet
import dataprocessing.arkitscenes as arkitscenes
import dataprocessing.s3dis as s3dis
import os
from glob import glob
import numpy as np
import pickle as pkl
from tqdm import tqdm
from models import model
from utils.util import convertSecs
from utils.eval_metric import compute_eval, save_results, save_pr_curves
from torch.utils.tensorboard import SummaryWriter
import sys
import open3d as o3d
from utils.util import colors, to_color, get_bbs_lines
import scipy.stats as stats
import pyviz3d.visualizer as viz
import uuid
import utils.s3dis_util as s3dis_util
from matplotlib import cm
from utils.util import *
from models.iou_nms import *

import utils.box_util as box_util
import utils.evaluate_detections as evaluate_detections
from scipy.spatial import ConvexHull
import math
import quaternion
import json

class Evaluater(object):
    def __init__(self, model, cfg, device=torch.device("cuda"), closest_to= None):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cfg = cfg
        _, _, self.ckpt_name, self.iteration_num = self.model.load_checkpoint(cfg.checkpoint, closest_to=closest_to)
        self.results_path = os.path.join(cfg.exp_path, 'results', self.ckpt_name)

        if self.cfg.dataset_name == 'scannet':
            self.semantic_valid_class_ids = scannet.SCANNET_SEMANTIC_VALID_CLASS_IDS
            self.semantic_id2idx = scannet.SCANNET_SEMANTIC_ID2IDX
            self.instance_id2idx = scannet.SCANNET_INSTANCE_ID2IDX
            self.is_foreground = scannet.is_foreground
        elif self.cfg.dataset_name == 'arkitscenes':
            self.semantic_valid_class_ids = arkitscenes.ARKITSCENES_SEMANTIC_VALID_CLASS_IDS
            self.semantic_id2idx = arkitscenes.ARKITSCENES_SEMANTIC_ID2IDX
            self.instance_id2idx = arkitscenes.ARKITSCENES_INSTANCE_ID2IDX
            self.is_foreground = arkitscenes.is_foreground
        elif self.cfg.dataset_name == 's3dis':
            self.semantic_valid_class_ids = s3dis.S3DIS_SEMANTIC_VALID_CLASS_IDS
            self.semantic_id2idx = s3dis.S3DIS_SEMANTIC_ID2IDX
            self.instance_id2idx = s3dis.S3DIS_INSTANCE_ID2IDX
            self.is_foreground = s3dis.is_foreground
        
        os.makedirs(self.results_path, exist_ok=True)

    def get_predictions_path(self, batch_size = 1):
        return os.path.join(
            self.results_path,
            f'batches_w_predictions{("@bs" + str(batch_size) if batch_size is not None else "")}'
            f'{("@seed" + str(self.cfg.fixed_seed) if self.cfg.fixed_seed is not None else "")}.pkl')

    # predict all results for a full dataset
    def dataset_prediction(self, dataset, dump = False, batch_size = None, first_n = None, random_n = None):
        shuffle = False
        if random_n:
            shuffle = True
            first_n = random_n
        data_iter = dataset.get_loader(shuffle=shuffle, drop_last=False, batch_size=batch_size).__iter__()
        batches = []
        predictions = []

        for i, batch in tqdm(enumerate(data_iter)):
            if first_n is not None and i > first_n:
                break

            import copy
            batch_cp = copy.deepcopy(batch)  # needed because of issues of the pytorch dataloader
            del batch
            predictions.append(self.model.get_prediction(batch_cp))
            batches.append(batch_cp)
        if dump:
            with open(self.get_predictions_path(batch_size), "wb") as outfile:
                pkl.dump((batches, predictions), outfile)
        return batches, predictions

    # converts a set of batches with their predictions to final result format: mask, score, semantics
    def dataset_pred2result(self, batches, predictions):
        results = {}
        for i in tqdm(range(len(batches))):
            results.update(self.model.pred2mask(batches[i], predictions[i], 'eval'))
        return results

    # do prediction and evaluation on all scenes of a dataset
    def eval(self, val_dataset, write_to_tb = False):
        print('Get predictions on full dataset:')
        if self.cfg.dataset_name == 's3dis':
            return self.s3dis_eval (val_dataset)

        if os.path.exists(self.get_predictions_path()):
            print('Loading dumped predictions...')
            with open(self.get_predictions_path(), "rb") as input_file:
                batches, predictions = pkl.load(input_file)
            if self.cfg.fixed_seed:
                print('Warning: fixed seed ignored when loading the precomputed data!')
        else:
            batches, predictions = self.dataset_prediction(val_dataset, batch_size=1)
        print (len (batches), type (batches))
        print('Convert predictions to mask format:')
        results = self.dataset_pred2result(batches, predictions)
        print (len (batches), type (batches))

        if self.cfg.dataset_name == 'scannet':
            return self.scannet_eval (results, write_to_tb)
        if self.cfg.dataset_name == 'arkitscenes':
            return self.arkitscenes_eval (results, batches, predictions)
        
    def s3dis_eval (self, val_dataset, viz_path=None, visualize_only=False):
        from utils.s3dis_util import clustering_for_background, assign_semantics_to_proposals, visualize_prediction

        model = self.model
        cfg = self.cfg
        scene_names = s3dis.get_scene_names ('val', cfg)
        
        val_iter = val_dataset.get_loader(shuffle=False, drop_last=False, batch_size=1)
        cluster_th, score_th, mask_bin_th, mask_nms_th  = cfg.eval_ths
        folds = [cfg.s3dis_split_fold]
        gt_labels = []
        pred_labels = []

        for iter, batch in enumerate (val_iter):
        
            prediction = model.get_prediction(batch, with_grad = False, to_cpu = True, min_size = True)
            scene = batch ['scene'][0]
            labels = batch ['labels'][0]

            scene_name = scene ["name"]
            print ("processing scene", iter, "-", scene_name)
            vox_pred_semantics = np.argmax (prediction ['mlp_per_vox_semantics'].cpu ().numpy (), 1)

            scene = batch ['scene'][0]
            labels = batch ['labels'][0]
            scene_name = scene ['name']

            if cfg.full_resolution:
                cfg.point_sampling_rate = None
                scene_full, labels_full = s3dis.process_scene (scene_name, 'val', cfg)
                sparse2dense = get_sparse2dense (scene_full, scene, cfg)

            results = self.model.pred2mask(batch, prediction, mode='eval')
            
            gt_label = {}
            gt_label ["semantics"] = labels ["semantics"]
            gt_label ["instances"] = labels ["instances"]
            pred_label = {}

            vox_id_per_point = batch["vox2point"][0]
            pred_semantics = vox_pred_semantics [vox_id_per_point]

            pred_label ["semantics"] = pred_semantics
            prediction ["vox_semantics"] = vox_pred_semantics
            prediction ["pred_semantics"] = pred_label ["semantics"]

            # Clustering wall / floor / ceiling differently
            background_pred_instances = clustering_for_background (pred_semantics, scene["positions"], scene["normals"])
            # Using majoring vote to assign semantic prediction for each point
            proposal_semantics = assign_semantics_to_proposals (pred_semantics, results [scene_name]['mask'])
            prediction ["pred_prop_semantics"] = proposal_semantics

            # Initialize instance predictions to be all -1 (ignored in the evaluation code)
            pred_instances = np.zeros_like (labels ["instances"]) - 1
            for idx, prop_mask in enumerate (results [scene_name]['mask']):
                unlabeled_mask = pred_instances < 0
                original_point_count = np.count_nonzero (prop_mask > 0)
                if proposal_semantics [idx] < 3:
                    continue
                # Exclude the points that were assigned a label earlier
                prop_mask = (prop_mask > 0) & unlabeled_mask
                # Not updating segments that have only 0.6 points of its original size or fewer
                filtered_point_count = np.count_nonzero (prop_mask)
                if (1.0 * filtered_point_count / original_point_count < 0.6):
                    continue
                # Ignore proposals that have less than 200 points
                if filtered_point_count < 200:
                    continue
                pred_instances [(prop_mask > 0)] = idx + 1
                prediction ["pred_semantics"][(prop_mask > 0)] = proposal_semantics [idx]

            prediction ["pred_semantics"] = pred_label ["semantics"]
            # Assign background instances to the final prediction with highest confidence score (always overwrite existing assignment)
            max_id_wo_bg = np.max (pred_instances)
            background_pred_instances [background_pred_instances > 0] += max_id_wo_bg
            pred_instances [background_pred_instances > 0] = background_pred_instances [background_pred_instances > 0]
            for class_id in range (13):
                class_mask = pred_label ["semantics"] == class_id
                prop_ids, prop_cnts = np.unique (pred_instances[class_mask], return_counts=True)
                id_small_mask = prop_cnts < 200
                small_prop_ids = prop_ids [id_small_mask]
                small_mask = np.isin (pred_instances[class_mask], small_prop_ids)

                tmp = pred_instances[class_mask]
                tmp [small_mask] = -1

                pred_instances[class_mask] = tmp

            pred_label ["instances"] = pred_instances
            if not cfg.full_resolution:
                gt_labels.append (gt_label)
                pred_labels.append (pred_label)
            else:
                gt_label ["semantics"] = labels_full ["semantics"]
                gt_label ["instances"] = labels_full ["instances"]
                pred_label ["semantics"] = pred_label ["semantics"][sparse2dense]
                pred_label ["instances"] = pred_label ["instances"][sparse2dense]
                gt_labels.append (gt_label)
                pred_labels.append (pred_label)

            if viz_path is not None:
                viz_path = os.path.join (viz_path, scene_name)
                os.makedirs (viz_path, exist_ok=True)
                visualize_prediction (self.cfg, scene_name, scene, labels, pred_label, viz_path)

        if not visualize_only:
            # Calculate precision and recall
            mPrecicision, mRecall, precisions, recalls = s3dis_util.s3dis_eval (pred_labels, gt_labels)
            print ("mean Precision", mPrecicision)
            print ("mean Recall", mRecall)
            print ("Precision of each class")
            print ("\tClass name \t\t Precision")
            for name, prec in zip (s3dis.ID2NAME, precisions):
                print(f'{name:>15}: \t {prec:.3f}')
            print ("recall of each class")
            print ("\tClass name \t\t Recall")
            for name, rec in zip (s3dis.ID2NAME, recalls):
                print(f'{name:>15}: \t {rec:.3f}')
                
    
    # do prediction and evaluation on all scenes of a dataset
    def arkitscenes_eval (self, results, batches, predictions, oriented_boxes=True, iou_t=0.5, visualize=False):
        pred_all = {}
        gt_all = {}
        print (len (batches), type (batches))
        batch = batches [0]
        for i in range(len(batches)):
            batch = batches[i]
            result = results[list(results.keys())[i]]

            scene = batch['scene'][0]
            labels = batch['labels'][0]

            # Prepare ground truth bounding boxes for evaluation
            groundtruth_list = []
            for i in range(labels['per_instance_bb_centers'].shape[0]):
                bounds = labels['per_instance_bb_bounds'][i]
                rotation_matrix = np.reshape(labels['per_instance_bb_rotations'][i], [3, 3]).T
                bounding_box_center = labels['per_instance_bb_centers'][i]
                if oriented_boxes:
                    bounding_box = box_util.get_oriented_corners(bounds, rotation_matrix, bounding_box_center)
                else:
                    bounding_box_size = box_util.get_rotated_bounds(bounds, rotation_matrix) * 2.0
                    bounding_box = np.array(np.concatenate([bounding_box_center, bounding_box_size], axis=0))
                groundtruth_list.append([labels['per_instance_semantics'][i], bounding_box])

            # Prepare predicted bounding boxes for evaluation
            predictions_list = []
            if visualize:
                v = viz.Visualizer()
                v.add_points('points', scene['positions'], scene['colors'] * 255, scene['normals'], point_size=10)
            for i in range(result['label_id'].shape[0]):
                label_class = result['label_id'][i]
                score = result['conf'][i]

                positions = scene['positions'][result['mask'][i]]
                if positions.shape[0] < 50:
                    continue
                # rotate(positions)
                if oriented_boxes:
                    points_2d = positions[:, 0:2]
                    convex_hull_vertices_2d = points_2d[ConvexHull(points_2d).vertices]
                    points_z_min, points_z_max = np.min(positions[:, 2]), np.max(positions[:, 2])

                    convex_hull_3d_bottom = np.concatenate(
                        [convex_hull_vertices_2d, np.ones([convex_hull_vertices_2d.shape[0], 1]) * points_z_min], axis=1)
                    convex_hull_3d_top = np.concatenate(
                        [convex_hull_vertices_2d, np.ones([convex_hull_vertices_2d.shape[0], 1]) * points_z_max], axis=1)
                    convex_hull_3d = np.concatenate([convex_hull_3d_bottom, convex_hull_3d_top], axis=0)
                    bounding_box = convex_hull_3d

                else:
                    positions_min, positions_max = np.min(positions, axis=0), np.max(positions, axis=0)
                    bounding_box_center = (positions_min + positions_max) / 2.0
                    bounding_box_size = (positions_max - positions_min)
                    bounding_box = np.concatenate([bounding_box_center, bounding_box_size], axis=0)
                predictions_list.append([label_class, bounding_box, score])

            pred_all[scene['name']] = predictions_list
            gt_all[scene['name']] = groundtruth_list

        if oriented_boxes:
            iou_func = evaluate_detections.get_iou_obb
        else:
            iou_func = evaluate_detections.get_iou

        rec, prec, ap = evaluate_detections.eval_det(pred_all, gt_all, get_iou_func=iou_func, ovthresh=iou_t)

        for k, v in sorted(ap.items()):
            print(f'{arkitscenes.arkitscenes_name_from_semantic_class_id[k]:>15}: \t {v:.3f}')
        mAP = np.mean(np.array([v for k, v in ap.items() if not math.isnan(v)]))
        print("mAP: ", mAP)
        return mAP

    def scannet_eval (self, results, write_to_tb):
        # compute evaluation on the results
        print('Compute eval metrics:')
        avgs, pr_curves = compute_eval(results)

        eval_folder = self.results_path + \
                      f"{'param_search' if self.cfg.eval_specific_param else ''}/mAP50_{avgs['all_ap_50%']:.3f}_eval"
        eval_folder += f"_ths:{'_'.join(str(th) for th in self.cfg.eval_ths)}"

        if self.cfg.fixed_seed:
            eval_folder += f'_seed:{self.cfg.fixed_seed}'
        else:
            eval_folder += f'_rid:{str(uuid.uuid1())[:8]}'

        if self.cfg.eval_wo_aug:
            eval_folder += f'_wo_aug{"_align" if self.cfg.align else ""}'

        print('Save results to disk.')
        os.makedirs(eval_folder, exist_ok=True)
        # write out results table
        save_results(avgs, eval_folder)
        # write out PR-curves as graphs
        save_pr_curves(pr_curves, eval_folder)

        # Write scores to tensorboard
        ap_all, ap_50, ap_25 = avgs["all_ap"], avgs["all_ap_50%"], avgs["all_ap_25%"]
        if write_to_tb:
            writer = SummaryWriter(os.path.dirname(__file__) +
                                        '/../experiments/tf_summaries/{}/'.format(self.cfg.exp_name))
            for ap_str, ap in [('ap_all', ap_all), ('ap_50', ap_50), ('ap_25', ap_25)]:
                writer.add_scalar('val/' + ap_str, ap, self.iteration_num)
            writer.close()
        return ap_all, ap_50, ap_25

    # parameter search (run on cpu)
    def param_search(self, val_dataset):
        if not os.path.exists(self.get_predictions_path()):
            self.dataset_prediction(val_dataset, batch_size=1, dump=True)
        args = sys.argv[1:]
        args.remove('--param_search')
        args = (' ').join(args)
        for cluster_th in np.linspace(*self.cfg.cluster_th_search):
            for score_th in np.linspace(*self.cfg.score_th_search):
                for mask_bin_th in np.linspace(*self.cfg.mask_bin_th_search):
                    for mask_nms_th in np.linspace(*self.cfg.mask_nms_th_search):
                        eval_ths = f'{cluster_th:.3f} {score_th:.3f} {mask_bin_th:.3f} {mask_nms_th:.3f}'
                        cmd = 'python models/evaluation.py ' + args + f' --eval_device cpu --eval_specific_param --eval_ths ' + eval_ths
                        sbatch = f'sbatch -p cpu20 -t 03:00:00 --mem-per-cpu 100GB -o /BS/Open4D2/work/Slurm_Outs/%A.out --wrap="{cmd}" --job-name="{eval_ths}"'
                        os.system(sbatch)

    def produce_visualizations_arkitscenes (self, val_dataset):
        print('Get predictions:')
        batches, predictions = self.dataset_prediction(val_dataset, batch_size=1) 
        print('Convert predictions to mask format:')
        results = self.dataset_pred2result(batches, predictions)
        vis_folder = self.results_path + f"/viz/"
        cfg = self.cfg

        from dataprocessing.arkitscenes import arkitscenes_color_map

        pred_all = {}
        gt_all = {}
        for i in range(len(batches)):
            batch = batches[i]
            result = results[list(results.keys())[i]]

            item = batch

            scene_name = item['scene'][0]['name']
            positions = item['scene'][0]['positions']
            colors = item['scene'][0]['colors']
            normals = item['scene'][0]['normals']
            unique_instances = item['labels'][0]['unique_instances']
            per_instance_bb_centers = item['labels'][0]['per_instance_bb_centers']
            per_instance_bb_bounds = item['labels'][0]['per_instance_bb_bounds']
            per_instance_bb_rotations = item['labels'][0]['per_instance_bb_rotations']
            per_instance_semantics = item['labels'][0]['per_instance_semantics']

            # per segment
            num_points = positions.shape[0]
            semantics_per_point = np.zeros(num_points, dtype=int)
            for segment_id in range(item['gt_semantics'].shape[0]):
                semantics_per_point[item['seg2point'][0] == segment_id] = item['gt_semantics'][segment_id]
            colors_sem = arkitscenes.arkitscenes_color_map[np.squeeze(semantics_per_point)]

            # Compute mask - iteratve over all GT boxes

            bb_min_z, bb_max_z = 100, -100
            for i in unique_instances.tolist():
                if bb_min_z > per_instance_bb_centers[i, 2] - per_instance_bb_bounds[i, 2]:
                    bb_min_z = per_instance_bb_centers[i, 2] - per_instance_bb_bounds[i, 2]
                if bb_max_z < per_instance_bb_centers[i, 2] + per_instance_bb_bounds[i, 2]:
                    bb_max_z = per_instance_bb_centers[i, 2] + per_instance_bb_bounds[i, 2]
            positions -= np.array([0, 0, bb_min_z])

            upper_limit = 2.5
            mask = np.logical_and(positions[:, 2] < upper_limit, positions[:, 2] > (bb_min_z - 0.50))
            mask2 = np.logical_and(mask, semantics_per_point > 0)

            col = arkitscenes.arkitscenes_color_map[1:, :]
            col = np.concatenate([col, col * 1.5], axis=0)
            np.random.shuffle(col)

            v = viz.Visualizer()
            per_instance_bb = np.zeros([per_instance_bb_bounds.shape[0], 3 + 3 + 9 + 3], dtype=np.float32)
            for i in unique_instances.tolist():
                rotation_matrix = np.reshape(per_instance_bb_rotations[i], [3, 3]).T
                rotation_quaternion = quaternion.from_rotation_matrix(rotation_matrix)
                per_instance_bb[i, 0:3] = per_instance_bb_centers[i, :] - np.array([0, 0, bb_min_z])
                per_instance_bb[i, 3:6] = per_instance_bb_bounds[i, :]
                per_instance_bb[i, 6:15] = np.reshape(rotation_matrix, [1, -1])
                per_instance_bb[i, 15:18] = arkitscenes.arkitscenes_color_map[per_instance_semantics[i]] / 255.0

            # show predicted masks
            semantic_colors = np.ones_like(positions, int) * 255
            instance_colors = np.ones_like(positions, int) * 255
            fg_mask = np.zeros(positions.shape[0], bool)
            for i in range(result['mask'].shape[0]):
                inst_mask = result['mask'][i]
                label_id = result['label_id'][i]
                label_color = arkitscenes.arkitscenes_color_map[label_id]
                semantic_color = col[i % col.shape[0]]
                instance_colors[inst_mask, :] = semantic_color
                semantic_colors[inst_mask, :] = label_color
                fg_mask[inst_mask] = True
            
            out_path = os.path.join(vis_folder ,scene_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            # Load the original mesh
            mesh_path = os.path.join(cfg.data_dir, '3dod/Validation', scene_name, scene_name+'_3dod_mesh.ply')
            mesh1 = o3d.io.read_triangle_mesh(mesh_path).compute_vertex_normals()
            # Perform same transformation as the dataloader is doing
            translation_xy = np.mean(np.array(mesh1.vertices)[:, 0:2], 0)
            translation_z = np.min(np.array(mesh1.vertices)[:, 2])
            mesh2 = mesh1.translate([-translation_xy[0], -translation_xy[1], - (translation_z + bb_min_z)])  

            oversegmentation_path = os.path.join(cfg.data_dir, '3dod/segmented_val_clean',
                                                scene_name + '_3dod_mesh.0.010000.segs.json')
            with open(oversegmentation_path, 'r') as f:
                segment_id_per_point = json.load(f)['segIndices']

            def hash_vec(vec):
                return vec[0] + 1000 * (vec[1] + 1000 * vec[2])

            interpolated_scene_semantics = copy.deepcopy(mesh2)
            interpolated_scene_instances = copy.deepcopy(mesh2)
            np.asarray(interpolated_scene_semantics.vertex_colors)[:, 0:3] = np.array([1.0, 1.0, 1.0])
            np.asarray(interpolated_scene_instances.vertex_colors)[:, 0:3] = np.array([1.0, 1.0, 1.0])
            segment_id_per_point = np.array(segment_id_per_point)

            print('interpolating...')
            for segment_id in set(segment_id_per_point):
                segment_mask = (segment_id_per_point == segment_id)
                segment_mask_sampled = segment_mask[::cfg.subsample_rate]
                if not np.any(fg_mask[segment_mask_sampled]):
                    continue
                if np.count_nonzero(fg_mask[segment_mask_sampled]) < np.count_nonzero(segment_mask_sampled) // 2:
                    continue
                segment_semantic_colors = semantic_colors[segment_mask_sampled]
                segment_instance_colors = instance_colors[segment_mask_sampled]
                di = {}
                max_count = -100
                most_popular_id = -1
                for point_id in range(segment_semantic_colors.shape[0]):
                    try:
                        di[hash_vec(segment_semantic_colors[point_id])] += 1
                        if di[hash_vec(segment_semantic_colors[point_id])] > max_count:
                            max_count = di[hash_vec(segment_semantic_colors[point_id])]
                            most_popular_id = point_id
                    except KeyError:
                        di[hash_vec(segment_semantic_colors[point_id])] = 1
                segment_semantic_color = segment_semantic_colors[most_popular_id]
                segment_instance_color = segment_instance_colors[most_popular_id]
                np.asarray(interpolated_scene_semantics.vertex_colors)[segment_mask, :] = segment_semantic_color / 255.0
                np.asarray(interpolated_scene_instances.vertex_colors)[segment_mask, :] = segment_instance_color / 255.0

            # Crop the upper part of the scene, so we can see inside it in blender
            crop_box = o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, upper_limit])
            mesh_cropped = mesh2.crop(crop_box)
            interpolated_scene_semantics = interpolated_scene_semantics.crop(crop_box)
            interpolated_scene_instances = interpolated_scene_instances.crop(crop_box)

            # Write RGB scene to disk
            o3d.io.write_triangle_mesh(os.path.join(out_path, f'{scene_name}_rgb.ply'), mesh_cropped)
            np.save(os.path.join(out_path, f'{scene_name}_bbs.npy'), per_instance_bb)

            o3d.io.write_triangle_mesh(os.path.join(out_path, f'{scene_name}_rgb.ply'), mesh_cropped)
            o3d.io.write_triangle_mesh(os.path.join(out_path, f'{scene_name}_instances.ply'), interpolated_scene_instances)
            o3d.io.write_triangle_mesh(os.path.join(out_path, f'{scene_name}_semantics.ply'), interpolated_scene_semantics)

            v.add_points(f'Input Scene', np.array(mesh_cropped.vertices), np.array(mesh_cropped.vertex_colors) * 255, np.array(mesh_cropped.vertex_normals), point_size=25, visible=True)
            v.add_points(f'Object Semantics', np.array(interpolated_scene_semantics.vertices), np.array(interpolated_scene_semantics.vertex_colors) * 255, np.array(interpolated_scene_semantics.vertex_normals), point_size=25, visible=False)
            v.add_points(f'Object Instances', np.array(interpolated_scene_instances.vertices), np.array(interpolated_scene_instances.vertex_colors) * 255, np.array(interpolated_scene_instances.vertex_normals), point_size=25, visible=False)

            v.save(os.path.join(out_path, "pyviz3d"))

            print('Done')
    
    def produce_visualizations_s3dis (self, val_dataset):
        vis_folder = self.results_path + f"/viz/"
        self.s3dis_eval (val_dataset, viz_path=vis_folder, visualize_only=True)
        

    def produce_visualizations_scannet (self, val_dataset):
        print('Get predictions:')
        batches, predictions = self.dataset_prediction(val_dataset, batch_size=1) 
        print('Convert predictions to mask format:')
        results = self.dataset_pred2result(batches, predictions)
        vis_folder = self.results_path + f"/viz/"

        from dataprocessing.scannet import scannet_color_map

        for batch in batches:
            for i, scene in enumerate(batch['scene']):

                out_path = os.path.join(vis_folder ,scene["name"])
                os.makedirs(out_path, exist_ok=True)

                # ---------------- WRITE OUT INPUT SCENE
                path_ply = f'/BS/impseg/work/ScanNet/scans/{scene["name"]}/{scene["name"]}_vh_clean_2.ply'
                mesh = o3d.io.read_triangle_mesh(path_ply)
                mesh.vertices = o3d.utility.Vector3dVector(scene['positions'])
                o3d.io.write_triangle_mesh(os.path.join(out_path,'rgb.ply'), mesh)

                # ---------------- WRITE OUT GT INSTANCES, SEMANTICS, and BBs
                labels = batch['labels'][i]
                color_map = cm.get_cmap('Paired', 12)
                colors_map = np.array(color_map(range(12)))[:, :3]
                r, g, b = colors_map.T
                colors_map = np.vstack((colors_map, np.array([r, b, g]).T, np.array([ b, r, g]).T, np.array([ g, r, b]).T, np.array([ b, g, r]).T, np.array([ g, b, r]).T))
                colors_map = np.vstack((colors_map,colors_map,colors_map))
                inst_colors = colors_map[labels['instances']]

                sem = labels['semantics']
                sem_colors = scannet_color_map[sem]
                point_invalid = ~np.isin(sem, self.semantic_valid_class_ids)
                sem_colors[point_invalid] = [200,200,200]
                sem_colors = sem_colors/255

                # remove bg colors (set it to white)
                point_scene_fg = (sem > 2) & (sem != 22)
                inst_colors[~point_scene_fg] = [1,1,1]
                sem_colors[~point_scene_fg] = [1,1,1]

                mesh.vertex_colors = o3d.utility.Vector3dVector(inst_colors)
                o3d.io.write_triangle_mesh(os.path.join(out_path,'gt_instances.ply'), mesh)
                mesh.vertex_colors = o3d.utility.Vector3dVector(sem_colors)
                o3d.io.write_triangle_mesh(os.path.join(out_path,'gt_semantics.ply'), mesh)

                # BBs
                bb_semantics = labels['per_instance_semantics']
                bb_fg = (bb_semantics > 2) & (bb_semantics != 22)
                bb_semantics = bb_semantics[bb_fg]
                bb_centers = labels['per_instance_bb_centers'][bb_fg]
                bb_bounds = 2 * labels['per_instance_bb_bounds'][bb_fg]
                bb_colors = scannet_color_map[bb_semantics]
                bb_invalid = ~np.isin(bb_semantics, self.semantic_valid_class_ids)
                bb_colors[bb_invalid] = [200,200,200]
                bbs = np.hstack((bb_centers, bb_bounds,bb_colors )).T
                np.save(os.path.join(out_path,'bbs'), bbs)

                # ---------------- WRITE OUT PREDICTED INSTANCES AND SEMANTICS
                pred_inst_colors = np.ones((len(mesh.vertices), 3)) * 255
                pred_sem_colors = np.ones((len(mesh.vertices), 3)) * 255
                for j, mask in enumerate(results[scene['name']]['mask']):
                    # maj vote
                    ins_id = stats.mode(labels['instances'][mask], None)[0][0]
                    if ins_id == 0:
                        # ins_id = len(labels['unique_instances']) + j
                        pred_inst_colors[mask] = [255,255,255]
                    else:
                        pred_inst_colors[mask] = colors_map[ins_id] * 255
                        # pred_inst_colors[mask] = np.array(color_map(ins_id))[:3] * 255
                    pred_sem_colors[mask] = scannet_color_map[results[scene['name']]['label_id'][j]]

                pred_sem_colors[point_invalid] = [200,200,200]
                mesh.vertex_colors = o3d.utility.Vector3dVector(pred_inst_colors / 255)
                o3d.io.write_triangle_mesh(os.path.join(out_path,'pred_instances.ply'), mesh)
                mesh.vertex_colors = o3d.utility.Vector3dVector(pred_sem_colors / 255)
                o3d.io.write_triangle_mesh(os.path.join(out_path,'pred_semantics.ply'), mesh)

                # -------------------- SAVE IN PYVIZ --------------------
                inst_colors *= 255
                sem_colors *= 255
                for colors_arr in [pred_inst_colors, pred_sem_colors, inst_colors, sem_colors]:
                    colors_arr[~point_scene_fg] = [100,100,100]
                v = viz.Visualizer()
                v.add_points(f'Input scene', scene['positions'], scene['colors'] * 255, point_size=25, normals = scene['normals'], visible=False)
                v.add_points(f'Pred Instances',scene['positions'], pred_inst_colors, point_size=25, normals = scene['normals'], visible=False)
                v.add_points(f'Pred Semantics',scene['positions'], pred_sem_colors, point_size=25,  normals = scene['normals'], visible=False)
                v.add_points(f'GT Instances',scene['positions'],inst_colors , point_size=25,  normals = scene['normals'], visible=False)
                v.add_points(f'GT Semantics',scene['positions'], sem_colors, point_size=25,  normals = scene['normals'], visible=False)
                start, end = get_bbs_lines(bb_centers, bb_bounds / 2)
                bbs_colors = np.repeat(bb_colors, 12, axis=0)
                v.add_lines(f'GT BBs', start, end, bbs_colors, visible=False)
                v.save(os.path.join(out_path,'pyviz3d'), verbose=False)

    def submission_write_out(self, mode):
        # Write output to submission format - only available for Scannet
        if self.cfg.dataset_name != 'scannet':
            return
        dataset = ScanNet(mode, self.cfg, do_augmentations=not self.cfg.eval_wo_aug)
        print('Get predictions:')
        batches, predictions = self.dataset_prediction(dataset, batch_size=1) 
        print('Convert predictions to mask format:')
        results = self.dataset_pred2result(batches, predictions)
        submission_folder = self.results_path + f"/submission_format"
        if self.cfg.fixed_seed:
            submission_folder += f'_seed:{self.cfg.fixed_seed}'
        else:
            submission_folder += f'_rid:{str(uuid.uuid1())[:8]}'
        if mode == 'test':
            submission_folder += f'_testset'
        mask_folder = submission_folder + '/predicted_masks'
        os.makedirs(submission_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        def export_ids(file_path, mask):
            with open(file_path, 'w') as f:
                for v in mask:
                    f.write('%d\n' % v)

        for batch in tqdm(batches):
            for i, scene in enumerate(batch['scene']):
                scene_txt = submission_folder + f'/{scene["name"]}.txt'
                with open(scene_txt, 'w') as f:
                    for j, mask in enumerate(results[scene['name']]['mask']):
                        label_id = results[scene['name']]['label_id'][j]
                        score = results[scene['name']]['conf'][j]
                        f.write('%s %d %f\n' % (f'predicted_masks/{scene["name"]}_{str(j)}.txt', label_id, score))
                        export_ids(mask_folder + f'/{scene["name"]}_{str(j)}.txt', mask)


if __name__ == '__main__':
    import torch
    from models.dataloader import ScanNet
    import config_loader as cfg_loader
    from models.evaluation import Evaluater

    print('start evaluation')
    cfg = cfg_loader.get_config()

    
    
    if cfg.dataset_name == 'scannet':
        import dataprocessing.scannet as scannet
        semantic_valid_class_ids_torch = scannet.SCANNET_SEMANTIC_VALID_CLASS_IDS_torch
        is_foreground = scannet.is_foreground
        semantic_id2idx = scannet.SCANNET_SEMANTIC_ID2IDX
        instance_id2idx = scannet.SCANNET_INSTANCE_ID2IDX

        if not cfg.predict_specific_scene:
            val_dataset = ScanNet('val', cfg, do_augmentations= not cfg.eval_wo_aug)
        else:
            val_dataset = ScanNet('predict_specific_scene', cfg, do_augmentations=False)
    elif cfg.dataset_name == 'arkitscenes':
        import dataprocessing.arkitscenes as arkitscenes
        if not cfg.predict_specific_scene:
            val_dataset = ARKitScenes('val', cfg, subsample_rate=cfg.subsample_rate)
        else:
            val_dataset = ARKitScenes('predict_specific_scene', cfg, subsample_rate=cfg.subsample_rate)
        semantic_valid_class_ids_torch = arkitscenes.ARKITSCENES_SEMANTIC_VALID_CLASS_IDS_torch
        semantic_id2idx = arkitscenes.ARKITSCENES_SEMANTIC_ID2IDX
        instance_id2idx = arkitscenes.ARKITSCENES_INSTANCE_ID2IDX
        is_foreground = arkitscenes.is_foreground
    elif cfg.dataset_name == 's3dis':
        import dataprocessing.s3dis as s3dis
        if not cfg.predict_specific_scene:
            val_dataset = S3DIS('val', cfg, do_augmentations=False)
        else:
            val_dataset = S3DIS('predict_specific_scene', cfg, do_augmentations=False)
        semantic_valid_class_ids_torch = s3dis.S3DIS_SEMANTIC_VALID_CLASS_IDS_torch
        semantic_id2idx = s3dis.S3DIS_SEMANTIC_ID2IDX
        instance_id2idx = s3dis.S3DIS_INSTANCE_ID2IDX
        is_foreground = s3dis.is_foreground

    model = model.Model(cfg, semantic_valid_class_ids_torch, semantic_id2idx, instance_id2idx, is_foreground, device=cfg.eval_device)

    # Evaluate the the checkpoints of previous 18 training days
    if cfg.eval_training: 
        for days in range(0,18):
            cfg_loader.set_fixed_seed(cfg)
            predictor = Evaluater(model, cfg, closest_to=(18-days)*24)
            predictor.eval(val_dataset, True)

    # Do a parameter search
    if cfg.param_search:
        predictor = Evaluater(model, cfg, closest_to = cfg.load_ckpt_closest_to)
        predictor.param_search(val_dataset)

    # Visualize the prediction
    if cfg.produce_visualizations or cfg.predict_specific_scene:
        predictor = Evaluater(model, cfg, closest_to = cfg.load_ckpt_closest_to)

        if cfg.dataset_name == 'scannet':
            predictor.produce_visualizations_scannet (val_dataset)
        elif cfg.dataset_name == 'arkitscenes':
            predictor.produce_visualizations_arkitscenes (val_dataset)
        elif cfg.dataset_name == 's3dis':
            predictor.produce_visualizations_s3dis (val_dataset)

    # Get the prediction in submission format for validation set
    if cfg.submission_write_out:
        predictor = Evaluater(model, cfg, closest_to = cfg.load_ckpt_closest_to)
        predictor.submission_write_out('val')

    # Get the prediction in submission format for test set
    if cfg.submission_write_out_testset:
        predictor = Evaluater(model, cfg, closest_to = cfg.load_ckpt_closest_to)
        predictor.submission_write_out('test')

    # Evaluate the validation set and output to console the scores
    if not (cfg.eval_training or cfg.param_search or cfg.produce_visualizations or cfg.submission_write_out or cfg.predict_specific_scene):
        predictor = Evaluater(model, cfg, device=cfg.eval_device, closest_to=cfg.load_ckpt_closest_to)
        predictor.eval(val_dataset, False)