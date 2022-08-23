
import torch
import models.detection_net as SelectionNet
import MinkowskiEngine as ME
import models.iou_nms as iou_nms
from scipy.stats import pearsonr
from models.iou_nms import *
from utils.util import *
from glob import glob
import os



class Model:
    def __init__(self, cfg, semantic_valid_class_ids, semantic_id2idx, instance_id2idx, is_foreground, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.semantic_valid_class_ids = semantic_valid_class_ids
        self.semantic_id2idx = semantic_id2idx
        self.instance_id2idx = instance_id2idx
        self.is_foreground = is_foreground
        self.detection_model = SelectionNet.SelectionNet(cfg, device, semantic_valid_class_ids, is_foreground, out_channels=[96, 96, 6]).to(device)
        if cfg.multigpu:
            self.detection_model = torch.nn.parallel.DistributedDataParallel(self.detection_model, device_ids=[device])
            self.detection_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.detection_model)
        # loss is computed by averaging over all element-wise computed loss entries
        # BCEWL includes sigmoid activation, needs un-normalized input
        self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss().to(device)
        # not used semantic labels are mapped to -100 using SEMANTIC_ID2IDX and ignored by this loss
        # CE is a softmax with exp activation, needs un-normalized inputs
        self.semantics_loss = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device)

    def compute_loss(self, batch, epoch):
        losses_dict, pred = self.compute_loss_detection(batch, epoch)

        return losses_dict

    def compute_loss_detection(self, batch, epoch):
        device = self.device
        cfg = self.cfg

        # transform data to voxelized sparse tensors
        sin = ME.SparseTensor(batch['vox_features'], batch['vox_coords'], device=device)

        # GET MODEL PREDICTION (and convert to regular pytorch tensors)
        pred = self.detection_model(sin, batch['pooling_ids'].to(device))

        # pred keys:
        #  mlp_offsets
        #  mlp_bounds
        #  mlp_bb_scores
        #  mlp_semantics
        #  vox_feats

        for mlp_head, sparse_tensor in pred.items():
            pred[mlp_head] = sparse_tensor.F

        # initialize loss
        losses_dict = {'optimization_loss': 0}

        # OFFSET loss (offset to BB center)
        if cfg.mlp_offsets in self.cfg.network_heads:
            # get gt and prediction
            gt_offsets, pred_offsets = batch['gt_bb_offsets'], pred[cfg.mlp_offsets]
            if self.cfg.loss_on_fg_instances or self.cfg.bb_supervision:
                pred_offsets = pred_offsets[batch['fg_instances']]
                gt_offsets = gt_offsets[batch['fg_instances']]

            # simple L1 loss over the predicted bounding box center offsets
            offset_loss_per_pred = torch.sum(torch.abs(pred_offsets - gt_offsets.to(device)), axis=1)
            offset_loss = torch.mean(offset_loss_per_pred)
            losses_dict['optimization_loss'] += self.cfg.loss_weight_bb_offsets * offset_loss
            losses_dict['offset_loss'] = offset_loss.detach()

        # BB size loss
        if cfg.mlp_bounds in self.cfg.network_heads:
            # get gt and prediction
            gt_bounds, pred_bounds = batch['gt_bb_bounds'], pred[cfg.mlp_bounds]
            if self.cfg.loss_on_fg_instances or self.cfg.bb_supervision:
                pred_bounds = pred_bounds[batch['fg_instances']]
                gt_bounds = gt_bounds[batch['fg_instances']]

            # simple L1 loss over the predicted bounding box bounds
            bounds_loss_per_pred = torch.sum(torch.abs(pred_bounds - gt_bounds.to(device)), axis=1)
            bounds_loss = torch.mean(bounds_loss_per_pred)

            losses_dict['optimization_loss'] += self.cfg.loss_weight_bb_bounds * bounds_loss
            losses_dict['bounds_loss'] = bounds_loss.detach()

        # Axis aligned bounding boxes IoU loss
        if cfg.use_bb_iou_loss:
            pred_bounds = pred[self.cfg.mlp_bounds]
            pred_offsets = pred[self.cfg.mlp_offsets]
            gt_bounds = batch['gt_bb_bounds']
            gt_offsets = batch['gt_bb_offsets']
            loc = batch['input_location']

            loc, gt_offsets, gt_bounds = loc.to(device), gt_offsets.to(device), gt_bounds.to(device)

            if self.cfg.loss_on_fg_instances or self.cfg.bb_supervision:
                pred_bounds = pred_bounds[batch['fg_instances']]
                pred_offsets = pred_offsets[batch['fg_instances']]
                gt_bounds = gt_bounds[batch['fg_instances']]
                gt_offsets = gt_offsets[batch['fg_instances']]
                loc = loc[batch['fg_instances']]

            pred_bounds = torch.clamp(pred_bounds, min=self.cfg.min_bb_size)  # enforce minimum size
            pred_bb_centers = pred_offsets + loc
            gt_bb_center = gt_offsets + loc
            pr_bbs = to_bbs_min_max_(pred_bb_centers, pred_bounds, device)
            gt_bbs = to_bbs_min_max_(gt_bb_center, gt_bounds, device)

            area1 = (pr_bbs[..., 3] - pr_bbs[..., 0]) * (pr_bbs[..., 4] - pr_bbs[..., 1]) * (pr_bbs[..., 5] - pr_bbs[..., 2])
            area2 = (gt_bbs[..., 3] - gt_bbs[..., 0]) * (gt_bbs[..., 4] - gt_bbs[..., 1]) * (gt_bbs[..., 5] - gt_bbs[..., 2])
            lt = torch.max(pr_bbs[..., :3], gt_bbs[..., :3])
            rb = torch.min(pr_bbs[..., 3:], gt_bbs[..., 3:])
            wh = (rb - lt).clamp(min=0)
            overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]
            union = area1 + area2 - overlap
            eps = 1e-6
            eps = union.new_tensor([eps])
            union = torch.max(union, eps)
            ious = overlap / union

            iou_loss_per_pred = 1.0 - ious
            iou_loss = torch.mean(iou_loss_per_pred)

            losses_dict['optimization_loss'] += self.cfg.loss_weight_bb_iou * iou_loss
            losses_dict['iou_loss'] = iou_loss.detach()


        # BB score loss
        if cfg.mlp_bb_scores in self.cfg.network_heads:
            loss_weight_bb_scores = self.cfg.loss_weight_bb_scores
            # hack because multi gpu needs to have the full network be part of loss computation already at beginning
            if epoch < self.cfg.mlp_bb_scores_start_epoch:
                loss_weight_bb_scores = 0
            # get gt and prediction
            pred_scores = pred[cfg.mlp_bb_scores].reshape(-1)  # (num_voxels)
            pred_bounds = pred[cfg.mlp_bounds]
            pred_offsets = pred[cfg.mlp_offsets]
            loc = batch['input_location']
            gt_offsets = batch['gt_bb_offsets']
            gt_bounds = batch['gt_bb_bounds']

            if self.cfg.loss_on_fg_instances or self.cfg.bb_supervision:
                pred_scores = pred_scores[batch['fg_instances']]
                pred_bounds = pred_bounds[batch['fg_instances']]
                pred_offsets = pred_offsets[batch['fg_instances']]
                loc = loc[batch['fg_instances']]
                gt_offsets = gt_offsets[batch['fg_instances']]
                gt_bounds = gt_bounds[batch['fg_instances']]

            loc, gt_offsets, gt_bounds = loc.to(device), gt_offsets.to(device), gt_bounds.to(device)

            # convert gt data to BB (min, max)-corner representation
            gt_bb_center = gt_offsets + loc
            gt_bbs = to_bbs_min_max_(gt_bb_center, gt_bounds, device)

            # convert pred data to BB (min,max)-corner representation
            pred_bounds = torch.clamp(pred_bounds, min=self.cfg.min_bb_size)  # enforce minimum size
            pred_bb_centers = pred_offsets + loc
            pred_bbs = to_bbs_min_max_(pred_bb_centers, pred_bounds, device)

            # compute IOU between pred and gt. This is the GT score that should be predicted.
            ious = iou_nms.set_IOUs(gt_bbs, pred_bbs).detach()  # (num_input_bbs)
            score_loss = self.BCEWithLogitsLoss(pred_scores, ious)

            # for interpretable logging, we use correlation
            corr, _ = pearsonr(ious.cpu().numpy(), pred_scores.cpu().detach().numpy())
            losses_dict['bb_scores_correlation'] = corr

            losses_dict['optimization_loss'] += loss_weight_bb_scores * score_loss
            losses_dict['bb_score_loss'] = score_loss.detach()
            # for test / visualization only
            losses_dict['bb_target_scores'] = torch.mean(ious)

        # center score loss
        if cfg.mlp_center_scores in self.cfg.network_heads and epoch >= self.cfg.mlp_center_scores_start_epoch:
            # get gt and prediction
            pred_scores = pred[cfg.mlp_center_scores].reshape(-1)  # (num_voxels)
            gt_scores = offset_loss_per_pred.detach()  # ( num_voxels)
            # simple L1 loss over the predicted scores
            if self.cfg.loss_on_fg_instances:
                pred_scores = pred_scores[batch['fg_instances']]
            score_loss = torch.abs(pred_scores - gt_scores)
            score_loss = torch.mean(score_loss)
            losses_dict['optimization_loss'] += self.cfg.loss_weight_center_scores * score_loss
            losses_dict['center_score_loss'] = score_loss.detach()
            # for interpretable logging, we use correlation
            corr, _ = pearsonr(gt_scores.cpu().numpy(), pred_scores.cpu().detach().numpy())
            losses_dict['center_scores_correlation'] = corr

        if self.cfg.mlp_semantics in self.cfg.network_heads:
            # get gt and prediction
            pred_semantics = pred[cfg.mlp_semantics]
            gt_semantics = batch['gt_semantics']
            # invalid and unlabeled ids are mapped to '-100' (the 'ignore'-label of our loss)
            gt_semantics = self.semantic_id2idx[gt_semantics].to('cuda')

            semantics_loss = self.semantics_loss(pred_semantics, gt_semantics)
            pred_semantics_int = torch.argmax(pred_semantics, 1)
            # this accuracy is pessimistic: it also measures unlabeled+invalid points
            semantics_acc = torch.sum(pred_semantics_int == gt_semantics) / len(gt_semantics)
            semantics_miou = semIOU(pred_semantics_int, gt_semantics).mean()

            losses_dict['optimization_loss'] += self.cfg.loss_weight_semantics * semantics_loss
            losses_dict['semantics_loss'] = semantics_loss.detach().cpu().numpy()
            losses_dict['semantics_acc'] = semantics_acc.detach().cpu().numpy()
            losses_dict['semantics_mIoU'] = semantics_miou

        return losses_dict, pred

    def get_prediction(self, batch, with_grad=False, to_cpu=True, min_size=True, get_all=False):

        pred = self.detection_model.get_prediction(batch, with_grad=with_grad, to_cpu=to_cpu, min_size=min_size)
        return pred

    def pred2mask(self, batch, pred, mode):
        return self.detection_model.detection2mask(batch, pred, self.cfg, mode,
                                                       True, *self.cfg.eval_ths)

    def parameters(self):
        return self.detection_model.parameters()

    def to(self, device):
        self.detection_model = self.detection_model.to(device)
        return self

    def eval(self):
        self.detection_model.eval()

    def train(self):
        self.detection_model.train()

    # returns if the checkpoint contained all parameters for the model
    def load_state_dict(self, state_dict, strict=True):
        if self.cfg.multigpu:
            return self.detection_model.module.load_state_dict(state_dict, strict)
        else:
            return self.detection_model.load_state_dict(state_dict, strict)


    def state_dict(self):

        if self.cfg.multigpu:
            return self.detection_model.module.state_dict()
        else:
            return self.detection_model.state_dict()

    def load_checkpoint(self, checkpoint=None, closest_to = None):
        checkpoints = glob(self.cfg.checkpoint_path + '/*')
        if checkpoint is None:
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.cfg.checkpoint_path))
                return 0, 0

            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)
            if closest_to:
                ckpt_idx = np.argmin(np.abs(checkpoints - (closest_to * 60 * 60)))
            else: #use last
                ckpt_idx = -1
            path = self.cfg.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
                *[*convertSecs(checkpoints[ckpt_idx]), checkpoints[ckpt_idx]])
        else:
            path = self.cfg.checkpoint_path + '{}.tar'.format(checkpoint)

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        return epoch, training_time, os.path.basename(path)[:-4], checkpoint['iteration_num']


if __name__ == '__main__':
    from models.dataloader import ScanNet
    import config_loader as cfg_loader

    device = torch.device('cuda')
    cfg = cfg_loader.get_config(['--config', 'configs/WKS.txt'])

    val_dataset = ScanNet('val', cfg)
    train_dataset = ScanNet('train', cfg)
    train_dataloader = train_dataset.get_loader()
    batch = next(iter(train_dataloader))

    model = Model(cfg)
    model.detection_model.load_checkpoint()
