import MinkowskiEngine as ME
import torch
from torch import nn
from models.resnet import ResNetBase, BasicBlock
from utils.util import *
from models.iou_nms import *
import random
import numpy as np

from tqdm import tqdm

class SelectionNet(ResNetBase):
    BLOCK = BasicBlock
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    added_PLANES = (256,256,256,256,256,256)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, cfg, device, semantic_valid_class_ids, is_foreground, out_channels=[96, 96, 3], D=3, mlp_head=False):
        self.device = device
        self.cfg = cfg
        self.mlp_head = mlp_head
        self.LAYERS = (cfg.layers,) * 8
        self.added_LAYERS = (cfg.layers,) * 6
        self.semantic_valid_class_ids = semantic_valid_class_ids
        self.is_foreground = is_foreground
        ResNetBase.__init__(self, cfg.in_channels, out_channels, D)


    def network_initialization(self, in_channels, out_channels = [96,96,3], D = None, mlp_head = False):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)  # out: 2 cm voxels

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)  # out: 4 cm voxels
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)  # out: 8 cm voxels
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)  # out: 16 cm voxels
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        # THIS WAS THE BOTTLENECK LAYER
        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)  # out: 32 cm voxels
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        ################### DEEPER MODEL ######################
        self.added_conv1p16s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)  # out: 64 cm voxels
        self.added_bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.added_block1 = self._make_layer(self.BLOCK, self.added_PLANES[0],
                                       self.added_LAYERS[0])

        self.added_conv2p32s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)  # out: 128 cm voxels
        self.added_bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.added_block2 = self._make_layer(self.BLOCK, self.added_PLANES[1],
                                       self.added_LAYERS[1])

        # NEW BOTTLENECK LAYER
        self.added_conv3p64s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)  # out: 256 cm voxels
        self.added_bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.added_block3 = self._make_layer(self.BLOCK, self.added_PLANES[2],
                                       self.added_LAYERS[2])

        # START OF UP CONV
        self.added_convtr4p128s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.added_PLANES[3], kernel_size=2, stride=2, dimension=D)
        self.added_bntr4 = ME.MinkowskiBatchNorm(self.added_PLANES[3])
        self.inplanes = self.added_PLANES[3] + self.added_PLANES[1] * self.BLOCK.expansion
        self.added_block4 = self._make_layer(self.BLOCK, self.added_PLANES[3],
                                       self.added_LAYERS[3])

        self.added_convtr5p64s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.added_PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.added_bntr5 = ME.MinkowskiBatchNorm(self.added_PLANES[4])
        self.inplanes = self.added_PLANES[4] + self.added_PLANES[0] * self.BLOCK.expansion
        self.added_block5 = self._make_layer(self.BLOCK, self.added_PLANES[4],
                                       self.added_LAYERS[4])

        self.added_convtr6p32s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.added_PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.added_bntr6 = ME.MinkowskiBatchNorm(self.added_PLANES[5])
        self.inplanes = self.added_PLANES[5] + self.PLANES[3] * self.BLOCK.expansion
        self.added_block6 = self._make_layer(self.BLOCK, self.added_PLANES[5],
                                       self.added_LAYERS[5])


        ################### DEEPER MODEL END ######################

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        # MLP for offset + bounds regression
        # below code is used instead of "mlp_head" function for backwards compatibility with checkpoint
        if self.cfg.load_unused_head: 
            self.final0 = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion,
                out_channels[0],
                kernel_size=1,
                bias=True,
                dimension=D)
            self.final0_bn = ME.MinkowskiBatchNorm(out_channels[0])

            self.final1 = ME.MinkowskiConvolution(
                out_channels[0],
                out_channels[1],
                kernel_size=1,
                bias=True,
                dimension=D)
            self.final1_bn = ME.MinkowskiBatchNorm(out_channels[1])

            self.final2 = ME.MinkowskiConvolution(
                out_channels[1],
                out_channels[2],
                kernel_size=1,
                bias=True,
                dimension=D)

        self.relu = ME.MinkowskiReLU()


        ## MLP for offset regression
        def mlp_head(output_dim):
            return nn.Sequential(
                ME.MinkowskiConvolution(
                    self.PLANES[7] * self.BLOCK.expansion,
                    out_channels[0],
                    kernel_size=1,
                    bias=True,
                    dimension=D),
                ME.MinkowskiReLU(),
                ME.MinkowskiBatchNorm(out_channels[0]),
                ME.MinkowskiConvolution(
                    out_channels[0],
                    out_channels[1],
                    kernel_size=1,
                    bias=True,
                    dimension=D),
                ME.MinkowskiReLU(),
                ME.MinkowskiBatchNorm(out_channels[1]),
                ME.MinkowskiConvolution(
                    out_channels[1],
                    output_dim,
                    kernel_size=1,
                    bias=True,
                    dimension=D)
            )

        self.network_heads = {}
        for network_head in self.cfg.network_heads:
            if network_head == self.cfg.mlp_offsets:
                self.mlp_offsets = mlp_head(3)
                self.network_heads[network_head] = self.mlp_offsets

            if network_head == self.cfg.mlp_bounds:
                self.mlp_bounds = mlp_head(3)
                self.network_heads[network_head] = self.mlp_bounds

            if network_head == self.cfg.mlp_bb_scores:
                self.mlp_score = mlp_head(1)
                self.mlp_score = mlp_head(1)
                self.network_heads[network_head] = self.mlp_score

            if network_head == self.cfg.mlp_center_scores:
                self.mlp_center_score = mlp_head(1)
                self.network_heads[network_head] = self.mlp_center_score


            if network_head == self.cfg.mlp_semantics:
                # Predict valid classes
                self.mlp_semantics = mlp_head(len(self.semantic_valid_class_ids))
                self.network_heads[network_head] = self.mlp_semantics


        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()



    def forward(self, x, pooling_ids = None):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        ################ DEEPER MODEL STARTS HERE
        # out = self.block4(out)
        out_b4p16 = self.block4(out)

        # tensor_stride=32
        out = self.added_conv1p16s2(out_b4p16)
        out = self.added_bn1(out)
        out = self.relu(out)
        out_added_b1p32 = self.added_block1(out)

        # tensor_stride=64
        out = self.added_conv2p32s2(out_added_b1p32)
        out = self.added_bn2(out)
        out = self.relu(out)
        out_added_b2p64 = self.added_block2(out)

        # NEW BOTTLENECK LAYER
        # tensor_stride=128
        out = self.added_conv3p64s2(out_added_b2p64)
        out = self.added_bn3(out)
        out = self.relu(out)
        out = self.added_block3(out)

        # tensor_stride=64
        out = self.added_convtr4p128s2(out)
        out = self.added_bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_added_b2p64)
        out = self.added_block4(out)

        # tensor_stride=32
        out = self.added_convtr5p64s2(out)
        out = self.added_bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_added_b1p32)
        out = self.added_block5(out)

        # tensor_stride=16
        out = self.added_convtr6p32s2(out)
        out = self.added_bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b4p16)
        out = self.added_block6(out)

        ################ END OF DEEPER MODEL

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out_b5p8 = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out_b5p8)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out_b6p4 = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out_b6p4)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out_b7p2 = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out_b7p2)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)


        outputs = {}

        if self.cfg.do_segment_pooling:
            assert pooling_ids is not None
            out.C[:,0] = pooling_ids
            out = ME.SparseTensor(out.F, out.C, device=self.device)
            if not self.cfg.max_pool_segments_detection_net:
                out = self.global_avg_pool(out)
            else:
                out = self.global_max_pool(out)


        for network_head in self.cfg.network_heads:
            outputs[network_head] = self.network_heads[network_head](out)

            if self.cfg.mlp_bounds_relu and network_head == self.cfg.mlp_bounds:
                outputs[network_head] = self.relu(outputs[network_head])

        return outputs

    # converts predictions to final result format: mask, score, semantics
    # higher cluster_th means more clusters, higher score_th means less masks, higher mask_bin_th means smaller masks
    # higher mask_nms_th means more masks
    def detection2mask(self, batch, pred, cfg, mode, score_filtering = True,
                       cluster_th = 0.3, score_th = 0.3, mask_bin_th = 0.3, mask_nms_th = 0.3):

        # we transform bb center + size format into min_xyz, max_xyz format
        pred_bbs = to_bbs_min_max(batch['input_location'].to('cuda' if pred['mlp_offsets'].is_cuda else 'cpu'),
                                  pred['mlp_offsets'], pred['mlp_bounds'],
                                  torch.nn.Sigmoid()(pred['mlp_bb_scores']))

        pred_semantics = pred[cfg.mlp_semantics]
        # get pred index
        pred_semantics = torch.argmax(pred_semantics, 1)
        # convert index to scannet id
        pred_semantics = self.semantic_valid_class_ids[pred_semantics].long()

        results = {}
        for scene_idx, scene in enumerate(batch['scene']):

            # get predictions that belong to the current scene
            scene_mask = batch['batch_ids'] == scene_idx

            scene_pred_semantics = pred_semantics[scene_mask]
        
            # predicted foreground
            # scene_pred_fg = (scene_pred_semantics > 2) & (scene_pred_semantics != 22)
            scene_pred_fg = self.is_foreground (scene_pred_semantics)
            scene_pred_bbs = pred_bbs.to('cpu')[scene_mask][scene_pred_fg]

            # ---------- Compute instance clusters ---------------
            # (output in same space as prediction, i.e. voxels or segments)
            # num_clusters, (num_clusters, num_elements), (num_clusters, num_fg_predictions)
            cluster_representatives, clusters, cluster_heatmaps \
                = NMS_clustering(scene_pred_bbs, cluster_th=cluster_th)

            scores = scene_pred_bbs[cluster_representatives][:, 0]
            scene_pred_bbs = scene_pred_bbs[cluster_representatives]
            # ---------- Filter clusters with too small pred quality
            if score_filtering:
                score_mask = scores > score_th
                cluster_heatmaps = cluster_heatmaps[score_mask]
                scores = scores[score_mask]
                scene_pred_bbs = scene_pred_bbs[score_mask]
                cluster_representatives = cluster_representatives[score_mask]

            # ---------- Remove duplicate masks with NMS ---------------
            # Done in vox space. Convert Heatmaps to voxel masks in case of segment prediction.
            if cfg.do_segment_pooling:
                # include background into masks via zero-padding and project them into voxel space
                pred_heatmaps_w_bg = torch.zeros(len(cluster_heatmaps), len(scene_pred_fg))
                pred_heatmaps_w_bg[:, scene_pred_fg] = cluster_heatmaps
                seg2vox = batch['seg2vox'][scene_idx]
                cluster_heatmaps = pred_heatmaps_w_bg[:, seg2vox] # num_voxels
                scene_pred_semantics = scene_pred_semantics[seg2vox]

            pred_masks = cluster_heatmaps > mask_bin_th
            mask_filter, _ = mask_NMS(pred_masks, mask_nms_th)
            pred_masks = pred_masks[mask_filter]
            bb_scores = scores[mask_filter]
            scene_pred_bbs = scene_pred_bbs[mask_filter]
            cluster_representatives = cluster_representatives[mask_filter]
            cluster_heatmaps = cluster_heatmaps[mask_filter]

            # ---------- Compute best matching semantic label per found instance
            # Also done in vox space.
            instance_labels = np.zeros(len(pred_masks), dtype='int32')
            for i, instance_mask in enumerate(pred_masks):
                # find most frequent semantic label within predicted instance
                counts = np.bincount(scene_pred_semantics[instance_mask])
                # assign most freq label to each mask
                instance_labels[i] = np.argmax(counts)

            # ----------- project predictions to point space
            if mode == 'eval':
                vox2point = batch['vox2point'][scene_idx]
                pred_masks = pred_masks[:,vox2point]

                results[scene['name']] = {
                                        'conf': bb_scores,  #(N_clusters, ) (tensor?)
                                        'label_id': instance_labels,  # (N_clusters, ) (np.array)
                                        'mask': pred_masks,  #(N_clusters, N_scene_points) [0 or 1] (tensor?)
                                        }
            else:
                results[scene['name']] = {
                                        'conf': bb_scores,  #(N_clusters, )
                                        'label_id': instance_labels,  # (N_clusters, )
                                        'mask': pred_masks,  #(N_clusters, N_voxels) [0 or 1]
                                        'cluster_representatives': cluster_representatives,
                                        'cluster_heatmaps': cluster_heatmaps, #(N_clusters, N_voxels) [0,1]
                                        'bbs': scene_pred_bbs,
                                        'pred_fg': scene_pred_fg
                                        }

        return results


    # uses SELF-model to predict data given a batch. allows to turn on/off back-propagation, and allows to apply
    # a minimum size to BB size predictions.
    def get_prediction(self, batch, with_grad = True, to_cpu = False, to_numpy = False, min_size = True):
        device = self.device

        if not with_grad:
            with torch.no_grad():
                # transform data to voxelized sparse tensors
                sin = ME.SparseTensor(batch['vox_features'], batch['vox_coords'], device=device)
                pred = self(sin, batch['pooling_ids'].to(device))
        else:
            sin = ME.SparseTensor(batch['vox_features'], batch['vox_coords'], device=device)
            pred = self(sin, batch['pooling_ids'].to(device))

        for mlp_head,sparse_tensor in pred.items():
            if not to_cpu:
                pred[mlp_head] = sparse_tensor.F
            else:
                pred[mlp_head] = sparse_tensor.F.cpu()
        if min_size:
            self.to_min_size(pred)

        if to_numpy:
            for mlp_head, tensor in pred.items():
                pred[mlp_head] = tensor.numpy()
        return pred

    def to_min_size(self, pred):
        if self.cfg.mlp_bounds in pred.keys() and self.cfg.min_bb_size is not None:
            pred[self.cfg.mlp_bounds] = torch.clamp(pred[self.cfg.mlp_bounds], min=self.cfg.min_bb_size)


    def load_checkpoint(self, load_idx=-1, checkpoint=None):
        from glob import glob
        import os
        checkpoints = glob(self.cfg.checkpoint_path + '/*')

        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.cfg.checkpoint_path))
            return
        if checkpoint is None:
            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)
            path = self.cfg.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
                *[*convertSecs(checkpoints[load_idx]), checkpoints[load_idx]])
        else:
            path = self.cfg.checkpoint_path + '{}.tar'.format(checkpoint)
        print('Loading checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
