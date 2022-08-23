import os

print('set OMP_NUM_THREADS=16, for the ME engine')
os.environ["OMP_NUM_THREADS"] = "16" 

import configargparse
import numpy as np
import random
import torch

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--exp_name", type=str, default=None,
                        help='Experiment name, used as folder name for the experiment.')
    parser.add_argument("--data_dir", type=str, default='./data/scannet/',
                        help='data directory')
    parser.add_argument("--data_split", type=str, default='data/scannet/scannetv2_official_split.npz',
                        help='Path to the data split file.')
    parser.add_argument("--dataset_name", type=str, default='scannet',
                        help='Name of the dataste. Options: {scannet, arkitscenes}')

    # --------------------- MODEL / DATA ----------------------------------------------------------------
    parser.add_argument("--num_workers", type=int, default=16,
                        help='Number of worker processes preparing input data.'
                             'The larger the better, but should not exceed the number of available CPUs.')
    parser.add_argument("--use_normals_input", default=False, action="store_true",
                            help="Use normal vectors as inputs for geometric encoding")

    # --------------------------- ARKitScenes -------------------------
    parser.add_argument("--subsample_rate", type=int, default=10,
                        help="subsample rate of the point cloud, arscenes are 10 times larger than scannet")


    # --------------------- BB SUPERVISION ----------------------------------------------------------------
    parser.add_argument("--bb_supervision", default=False, action="store_true",
                            help="If true, use only bounding boxes as supervision signal.")
    parser.add_argument("--point_association", default=False, action="store_true",
                            help="If true, get associations only based on points, if false, incorporate segments.")
    parser.add_argument("--smallest_bb_heuristic", default=False, action="store_true",
                            help="If true, get associations based on smallest bb.")
    parser.add_argument("--majority_vote", default=False, action="store_true",
                            help="If true, do majority vote per segment on point association data.")
    parser.add_argument("--dropout_boxes", type=float, default=None,
                        help="If set to a float value, represents probability to drop boxes. Default: no dropout.")
    parser.add_argument("--noisy_boxes", type=float, default=None,
                        help="If set to a float value, represents 2 times the std_dev used for sampling gaussian noise"
                             " to displace min_pt and max_pt of gt bounding boxes.")

    # -----------------------  DATA PROCESSING -----------------------------------------------------------
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="size of each voxel after the voxelization process of point cloud (eg. 0.02 is 2 cm)")

    parser.add_argument("--align", default=False, action='store_true',
                        help='Whether to align scenes to axis or not.')
    parser.add_argument("--dont_align", default=False, action='store_true',
                        help='Overwrites align.')

    parser.add_argument("--debug", default=False, action='store_true',
                        help='Compute only a few scenes, and save result visualization to disk.')
    parser.add_argument("--slurm_array_id", type=int,
                        help="Array ID of slurm for multi processing jobs.")

    # --------------------------- MULTI GPU --------------------------------------------------------------
    parser.add_argument("--multigpu", default=False, action='store_true',
                        help='Use multiple gpus.')
    parser.add_argument("--singlegpu", default=False, action='store_true',
                        help='Use single gpu (Default). Overwrites multigpu.')

    # --------------------------- DEBUGGING --------------------------------------------------------------
    parser.add_argument("--overfit_to_single_scene", type=int, default=None,
                        help='For debugging: Whether to train on a single scene. Defines index of scene to overfit to.')
    parser.add_argument("--overfit_to_single_scene_str", type=str, default=None,
                        help='For debugging: Whether to train on a single scene. Defines string of scene to overfit to.')
    parser.add_argument("--dataset_size", type=int, default=None,
                        help='For debugging: only use the specified number of samples from the dataset.')

    # -------------------------- EVAL/ PREDICTIOn ------------------------------------------------------------------
    parser.add_argument("--checkpoint", type=str, default=None,
                        help='Checkpoint that will be loaded for prediction/evaluation.')
    parser.add_argument("--fixed_seed", type=int, default=None,
                        help='Set a fixed seed for all random operations.')
    parser.add_argument("--sample_fixed_seed", default=False, action='store_true',
                        help="If true, a single random seed will be sampled and used as fixed_seed.")
    parser.add_argument("--predict_specific_scene", type=str, default=None, 
                        help="specify a scene from train or validation set to make a prediction and visualize")

    # ----------------- DETECTION NET - EVAL ---------------------------------
    parser.add_argument("--eval_ths", type=float, nargs=4, default=None,
                        help="Converting predictions to masks thresholds: "
                             "cluster_th, score_th, mask_bin_th, mask_nms_th. Used for detection net evaluations")
    parser.add_argument("--load_ckpt_closest_to", type=int, default=None,
                        help='For eval only: Load the checkpoint closest to the specified number of training hours.')
    parser.add_argument("--eval_training", default=False, action='store_true',
                        help="Eval multiple training checkpoints into tensorboard.")
    parser.add_argument("--produce_visualizations", default=False, action='store_true',
                        help="Save model predictions as visualizations.")
    parser.add_argument("--eval_device", type=str, default='cuda',
                        help='Device (cuda/cpu) to do evaluation on.')
    parser.add_argument("--eval_wo_aug", default=False, action='store_true',
                        help="Eval turning off all augmentations.")
    parser.add_argument("--submission_write_out", default=False, action='store_true',
                        help="Save results in submission format for ScanNet benchmark.")
    parser.add_argument("--submission_write_out_testset", default=False, action='store_true',
                        help="Save results in submission format for ScanNet benchmark.")
    parser.add_argument("--fig3", default=False, action='store_true',
                        help="Do visualizations for fig  3 in paper.")

    ### Param Search ---------------------------------
    parser.add_argument("--param_search", default=False, action='store_true',
                        help="Do param search of non-maximum-clustering.")
    parser.add_argument("--eval_specific_param", default=False, action='store_true',
                        help="Not a human interface. Only set via code.")

    parser.add_argument("--cluster_th_search", default=[0.3, 0.8, 6], nargs=3,
                        help='Input to np.linspace: min_val, max_val, num of equal intervals (incl. start and end).')
    parser.add_argument("--score_th_search", default=[0, 0.2, 5], nargs=3,
                        help='Input to np.linspace: min_val, max_val, num of equal intervals (incl. start and end).')
    parser.add_argument("--mask_bin_th_search", default=[ 0.2, 0.35, 4], nargs=3,
                        help='Input to np.linspace: min_val, max_val, num of equal intervals (incl. start and end).')
    parser.add_argument("--mask_nms_th_search", default=[ 0.4, 0.8, 5], nargs=3,
                        help='Input to np.linspace: min_val, max_val, num of equal intervals (incl. start and end).')


    # -------------------------- TRAINING ------------------------------------------------------
    parser.add_argument("--eval_first", dest='skip_first_eval', action='store_false',
                        help="If true, training code does initial evaluation when starting training.")
    parser.set_defaults(skip_first_eval=True)
    parser.add_argument("--eval_every", type=int, default=12,
                        help='Int: do full evaluation on validation set each X epoch.')
    parser.add_argument("--val_every", type=int, default=12,
                        help='Int: do 5 batch loss evaluation on validation set each X epoch.')
    parser.add_argument("--ckpt_every", type=int, default=4,
                        help='Int: save checkpoint every X epoch.')
    parser.add_argument("--train_submission", default=False, action='store_true',
                        help="For ScanNet submission. If true, training and validation dataset are used for training.")
    parser.add_argument("--loose_model_loading", default=False, action='store_true',
                        help="If true, model loading is not done in strict, exact key matching mode.")
    parser.add_argument("--load_unused_head", default=False, action='store_true',
                        help="For backwards compatibility - should be removed!")
    parser.add_argument("--apple_warmstart", default=False, action='store_true',
                        help="Warm start training, excluding mismatching semantic layer.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='Number of scene chunks provided to the NDF network in one batch during training.\
                            Influences training speed (larger batches result in shorter epochs) but also GPU \
                             memory usage (higher values need more memory). Needs to be balanced with \
                             num_sample_points_training')
    parser.add_argument("--num_epochs", type=int, default=1500,
                        help='Stopping citron for duration of training. Model converges much earlier: model convergence\
                         can be checked via tensorboard and is logged within the experiment folder.')
    parser.add_argument("--lr", type=float, default=1e-6,
                        help='Learning rate used during training.')
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help='Optimizer used during training.')

    parser.add_argument("--loss_on_all_instances", default=False, action='store_true',
                        help='Overwrites loss_on_fg_instances. Apply loss for bounding box bounds and offsets on all '
                             'instances, including BG.')

    # learning rate scheduler
    parser.add_argument("--use_lr_scheduler", default=False, action='store_true',
                        help="If ture, cosine LR scheduler is used, else none is used.")
    parser.add_argument("--lr_scheduler_start_epoch", type=int,
                        help='Epoch when scheduler starts changing the LR.')
    parser.add_argument("--lr_scheduler_end_epoch", type=int,
                        help='Epoch when LR = 0.')

    # ----------------   AUGMENTATIONS -----------------------------------------------------------
    parser.add_argument("--augmentation", default=False, action='store_true',
                        help="Use augmentations during training.")
    parser.add_argument("--position_jittering", type=float, default=[0.00, 0.01], nargs=2, metavar=['prob', 'sigma'],
                        help="Randomly translate the postions of each points in each dimension. \
                            The displacement distances are sampled from a Gaussion distribution with sigma as std variance")
    parser.add_argument("--scaling_aug", type=float, default=[0.0, 0.9, 1.1], nargs=3, metavar=('prob', 'min', 'max'),
                        help='Randomly scale up or down the scene with a scaling factor within [min, max]. \
                            Prob is the probability of using the augmentation.')
    parser.add_argument("--color_jittering_aug", default=[0.0, 0.1], type=float, nargs=2, metavar=('prob', 'jitter_range'),
                        help='Add a random noise within [-jitter_range, jitter_range] to each of the RGB channel. \
                            Prob is the probability of using the augmentation.')
    parser.add_argument("--HAIS_jitter_aug", default=False, action="store_true",
                        help="Point jittering of HAIS.")
    parser.add_argument("--rotation_aug", type=float, default=[0.0, np.pi / 100, 1], nargs="+",
                        help='We uniformly random sample one rotation (for  z-axis) within [-pi,pi] ,\
                                 one rotation (for x-axis) and one rotation for y-axis within [-max_xy_angle, max_xy_angle]. \
                                     Prob is the probability of using the augmentation.'
                             ' individual_prob defines probability to turn on/off z/x/y-rotation, each sampled'
                             ' individually.')
    parser.add_argument("--rotation_90_aug", default=False, action="store_true",
                        help="If true, we uniformly sample one rotation in 0,90,180,270 degree.")
    parser.add_argument("--mix_3d_color_aug", default=False, action="store_true",
                        help="If true, we use mix3d color augmentation and normalization.")
    parser.add_argument("--apply_hue_aug", default=False, action="store_true",
                        help="If true, we use mix3d color augmentation and normalization.")
    parser.add_argument("--flipping_aug", default=0.0, metavar='prob', type=float,
                        help='Randomly flip the scene along x or y axis. Prob is the probability of randomly flipping in each axis.')
    parser.add_argument("--elastic_distortion", default=0.0, metavar='prob', type=float,
                        help='Apply elastic distortion with the default parameters in the Spatio Temporal Segmentation work. Prob is the probability of using the augmentation')
    parser.add_argument("--elastic_distortion_HAIS", default=0.0, metavar='prob', type=float,
                        help='Slightly different parameter setting to "elastic_distortion".')
    parser.add_argument("--chromatic_auto_contrast", default=0.0, metavar='prob', type=float,
                        help="The probability to randomly blend the original color with a rescaled contrast color. \
                            Prob is the probability of using the augmentation.")
    parser.add_argument("--chromatic_translation", type=float, default=[0.0, 0.1], metavar=('prob', 'trans_range_ratio'), nargs=2,
                        help="Add random color to the image. Trans_range_ratio: ratio of translation i.e. 1.0 * 2 * ratio * rand(-0.5, 0.5). \
                            Prob is the probability of using the augmentation.")
    parser.add_argument("--random_brightness", type=float, default=[0.0, 0.1], metavar=('prob', 'factor_range'), nargs=2,
                        help="Randomly multiply the brighness by a factor that is between (1-factor_range, 1+factor_range).")


    # ----------------- DETECTION NET -----------------------------------
    parser.add_argument("--do_segment_pooling", default=False, action="store_true",
                        help="Boolean, indicating if we want to do prediction per segment instead of per voxel.")

    parser.add_argument("--network_heads", default=None, type=str, nargs="+",
                        choices=["mlp_offsets", "mlp_bounds", "mlp_bb_scores", "mlp_semantics", "mlp_center_scores"],
                        help="Lists of network heads. Possible values: mlp_offsets, mlp_bounds, mlp_bb_scores"
                             " mlp_semantics")
    parser.add_argument("--mlp_bounds_relu", default=False, action="store_true",
                        help="Boolean, indicating if we want to use relu activation for mlp_bounds")
    parser.add_argument("--max_pool_segments_detection_net", default=False, action="store_true",
                        help="Boolean, indicating if we want to use max pool instead of AVG pool over segments in "
                             "selection net model.")
    parser.add_argument("--layers", type=int, default=2,
                        help="Number of convolution layers in each u-net block. Default = 2")

    # ----------------- DETECTION NET - LOSSES --------------------------
    parser.add_argument("--use_bb_iou_loss", default=False, action="store_true",
                        help="If true we use IOU loss additionally to bb_offset and bb_bounds loss.")

    parser.add_argument("--loss_weight_semantics", type=float, default=None,
                        help="Weight applied to the loss on per voxel semantics prediction")
    parser.add_argument("--loss_weight_bb_offsets", type=float, default=1.0, 
                        help="Weight applied to the loss on the BB offsets.")
    parser.add_argument("--loss_weight_bb_bounds", type=float, default=None,
                        help="Weight applied to the loss on the BB bounds.")
    parser.add_argument("--loss_weight_bb_scores", type=float, default=None,
                        help="Weight applied to the bb score loss.")
    parser.add_argument("--loss_weight_center_scores", type=float, default=None,
                        help="Weight applied to the center score loss.")
    parser.add_argument("--loss_weight_bb_iou", type=float, default=None,
                        help="Weight applied to the bb iou loss.")

    parser.add_argument("--mlp_bb_scores_start_epoch", default=0, type=int,
                        help="Epoch, when training of mlp_bb_scores is started.")
    parser.add_argument("--mlp_center_scores_start_epoch", default=0, type=int,
                        help="Epoch, when training of mlp_center_scores is started.")


    parser.add_argument("--min_bb_size", default=0.04, type=float,
                        help="Minimum size of the bounding box side lengths. Set to 'None' for no post-processing.")

    return parser


def get_config(args=None):
    import os
    parser = config_parser()
    cfg = parser.parse_args(args)

    # to avoid configargparse bug
    if cfg.singlegpu:
        cfg.multigpu = False
    if cfg.dont_align:
        cfg.align = False
    cfg.loss_on_fg_instances = True
    if cfg.loss_on_all_instances:
        cfg.loss_on_fg_instances = False

    if len(cfg.rotation_aug) == 1:
        cfg.rotation_aug = [cfg.rotation_aug[0], np.pi / 100, 1]
    if cfg.sample_fixed_seed:
        random_data = os.urandom(4)
        cfg.fixed_seed = int.from_bytes(random_data, byteorder="big")

    if cfg.fixed_seed:
        set_fixed_seed(cfg)

    if cfg.dropout_boxes:
        assert 0 <= cfg.dropout_boxes <= 1

    # define variable names
    cfg.mlp_offsets = "mlp_offsets"
    cfg.mlp_bounds = "mlp_bounds"
    cfg.mlp_bb_scores = "mlp_bb_scores"
    cfg.mlp_center_scores = "mlp_center_scores"
    cfg.mlp_semantics = "mlp_semantics"
    cfg.network_heads_options = [cfg.mlp_offsets, cfg.mlp_bounds, cfg.mlp_bb_scores,
                                 cfg.mlp_semantics, cfg.mlp_center_scores]

    cfg.full_model = False
    if (cfg.mlp_bounds in cfg.network_heads and
     cfg.mlp_offsets in cfg.network_heads and
     cfg.mlp_semantics in cfg.network_heads and
     cfg.mlp_bb_scores in cfg.network_heads):
        cfg.full_model = True
    else:
        print('Warning: this is not a model allowing for instance segmentation.'
                ' mAP plotting during training is turned off.')
    cfg.in_channels = 3 + 3 * cfg.use_normals_input # RGB + normal (if use it)

    if cfg.exp_name == 'cfg_name':
        cfg_name = os.path.basename(cfg.config)
        assert cfg_name[-4:] == '.txt'
        cfg.exp_name = cfg_name[:-4]

    cfg.exp_path = os.path.dirname(__file__) + '/experiments/{}/'.format(cfg.exp_name)
    cfg.checkpoint_path = cfg.exp_path + 'checkpoints/'.format(cfg.exp_name)

    if cfg.mlp_center_scores in cfg.network_heads:
        assert cfg.mlp_offsetsßßßß

    assert set(cfg.network_heads) <= set(cfg.network_heads_options)

    # no duplicates
    assert len(np.unique(cfg.network_heads)) == len(cfg.network_heads)

    if cfg.batch_size == 1:
        print('WARNING: batch size is set 1. Our model is quiet deep and pools some of the smallest scenes to single'
              ' voxel. In that case an error in the batch normalization is likely. Set batch size > 1!')
    if cfg.mlp_bb_scores in cfg.network_heads and cfg.loss_weight_bb_scores is None:
        raise


    if cfg.mlp_semantics in cfg.network_heads:
        if not cfg.loss_weight_semantics:
            raise
    if cfg.use_bb_iou_loss:
        if not (cfg.mlp_offsets in cfg.network_heads and cfg.mlp_bounds in cfg.network_heads):
            raise
        if cfg.loss_weight_bb_iou is None:
            raise

    return cfg

def set_fixed_seed(cfg):
    torch.backends.cudnn.deterministic = True
    random.seed(cfg.fixed_seed)
    torch.manual_seed(cfg.fixed_seed)
    torch.cuda.manual_seed(cfg.fixed_seed)
    np.random.seed(cfg.fixed_seed)
    print(f'Fixed seed is: {cfg.fixed_seed}')

if __name__ == '__main__':
    print(get_config())