from __future__ import division


import sys
sys.path.append('.')

import torch, torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time
from collections import defaultdict
from pynvml import *
from utils.util import *
from models.model import Model
from models.evaluation import Evaluater
import torch.multiprocessing as mp
import torch.distributed as dist
import config_loader as cfg_loader
from models.dataloader import ScanNet
from models.dataloader import ARKitScenes
from models.dataloader import S3DIS

class Trainer(object):
    # set val_dataset to None if no validation should be performed
    def __init__(self, model, train_dataloader, val_dataset, cfg, rank = None):
        self.cfg = cfg
        self.model = model
        self.rank = rank
        self.main_process = not cfg.multigpu or rank == 0

        model_params = self.model.parameters()

        if cfg.optimizer == 'Adam':
            self.optimizer = optim.Adam(model_params, lr=cfg.lr)
        if cfg.optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(model_params)
        if cfg.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(model_params, momentum=0.9)

        self.epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_min = None
        self.writer = None
        if self.main_process:
            os.makedirs(self.cfg.checkpoint_path, exist_ok=True)
            self.writer = SummaryWriter(os.path.dirname(__file__) + '/../experiments/tf_summaries/{}/'.format(cfg.exp_name))
            # include copy of all variables and the configuration file into the experiments folder
            f = os.path.join(cfg.exp_path, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(cfg)):
                    attr = getattr(cfg, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            if cfg.config is not None:
                f = os.path.join(cfg.exp_path, 'config.txt')
                with open(f, 'w') as file:
                    file.write(open(cfg.config, 'r').read())

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.model.compute_loss(batch, self.epoch)
        optimization_loss = loss_dict['optimization_loss']
        optimization_loss.backward()
        self.optimizer.step()
        return loss_dict

    def train_model(self, epochs):
        start, training_time, iteration_num = self.load_checkpoint()

        iteration_start_time = time.time()
        for rel_epoch, epoch in enumerate(range(start, epochs)):
            if self.cfg.multigpu:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.epoch = epoch
            losses_epoch = defaultdict(int)  # default values are 0
            print(f'Start epoch {epoch}')

            if self.cfg.use_lr_scheduler:
                cosine_lr_after_step(self.optimizer, self.cfg.lr, epoch,
                                     self.cfg.lr_scheduler_start_epoch, self.cfg.lr_scheduler_end_epoch)
                if self.main_process:
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], iteration_num)
                    if epoch >= self.cfg.lr_scheduler_end_epoch:
                        print('END TRAINING --- LR scheduling end reached. Stop training.')
                        return

            time_prepare_training_batch_start = time.time()
            for batch_num, batch in enumerate(self.train_dataloader):

                # save model
                iteration_duration = time.time() - iteration_start_time

                #---------- EVALUATE and SAVE CHECKPOINT ------------------------#
                if self.main_process and batch_num == 0 and not self.cfg.skip_first_eval:
                    training_time += iteration_duration
                    iteration_start_time = time.time()

                    # save checkpoints after every ckpt_every epochs
                    if epoch % self.cfg.ckpt_every == 0:  # model is large, so save only now and then
                        print('Saving checkpoint...')
                        save_time = time.time()
                        self.save_checkpoint(epoch, training_time, iteration_num)
                        print(f'Done saving checkpoint ({str(time.time() - save_time)[:5]} s)')


                    val_losses = self.compute_val_loss(self.cfg.num_eval_batches)
                    print("VAL losses: {} ".format(val_losses))
                    # self.writer.add_scalars('Losses/val', val_losses, iteration_num)
                    for k, v in val_losses.items():
                        if not k[:11] == "mask_scores":
                            self.writer.add_scalar('val/' + k, v, iteration_num)
                        else:
                            self.writer.add_scalar('val_mask_scores/' + k[12:], v, iteration_num)
                    val_loss = val_losses['optimization_loss']

                    # Evaluation after every eval_every epochs
                    if self.val_dataset and (epoch % self.cfg.val_every == 0 or epoch % self.cfg.eval_every == 0):
                        print('start computing validation loss')
                        # free memory for validation computation
                        if 'loss_dict' in locals():  # just remove it if it exists
                            del loss_dict

                        # eval not needed for very early models, early models take long to eval
                        if epoch % self.cfg.eval_every == 0 and epoch >= 250 and self.cfg.full_model \
                                and self.cfg.dataset_name == 'scannet':

                            import dataprocessing.scannet as scannet
                            semantic_valid_class_ids_torch = scannet.SCANNET_SEMANTIC_VALID_CLASS_IDS_torch
                            is_foreground = scannet.is_foreground
                            semantic_id2idx = scannet.SCANNET_SEMANTIC_ID2IDX
                            instance_id2idx = scannet.SCANNET_INSTANCE_ID2IDX

                            val_model = Model(self.cfg, semantic_valid_class_ids_torch, semantic_id2idx, instance_id2idx, is_foreground)
                            predictor = Evaluater(val_model, self.cfg)
                            ap_all, ap_50, ap_25 = predictor.eval(val_dataset)
                            for ap_str, ap in [('ap_all', ap_all), ('ap_50', ap_50), ('ap_25', ap_25)] :
                                self.writer.add_scalar('val/' + ap_str, ap, iteration_num)

                        if self.val_min is None:
                            self.val_min = val_loss

                        if val_loss < self.val_min:
                            self.val_min = val_loss
                            for path in glob(self.cfg.exp_path + 'val_min=*'):
                                os.remove(path)
                            np.save(self.cfg.exp_path + 'val_min=checkpoint_{}h:{}m:{}s_{}.tar'
                                    .format(*[*convertSecs(training_time),training_time]), [epoch, iteration_num, val_loss])
                self.cfg.skip_first_eval = False

                # Compute time to prepare batch; this time is reset at the end of this for-loop
                time_prepare_training_batch_duration = time.time() - time_prepare_training_batch_start
                print(f'Time to prepare batch: {time_prepare_training_batch_duration:.3f}')
                if self.main_process:
                    self.writer.add_scalar('time/prepare_training_batch', time_prepare_training_batch_duration,
                                           iteration_num)

                # Optimize model
                time_training_step_start = time.time()
                loss_dict = self.train_step(batch)
                time_training_step_duration = time.time() - time_training_step_start
                if self.main_process:
                    self.writer.add_scalar('time/training_step', time_training_step_duration, iteration_num)

                # Transform losses to single values
                for k, v in loss_dict.items():
                    losses_epoch[k] += v.item()

                current_iteration = batch_num + epoch * len(self.train_dataloader)
                current_losses = {k: str(v.item())[:6] for k, v in loss_dict.items()}
                print(f'{current_iteration} dt:{time_training_step_duration:.3f} Current losses: {current_losses}')

                if self.main_process:
                    # LOGGING GPU STATISTICS
                    # in order to manage the unstable memory usage of ME (defined on GPU 0 here)
                    nvmlInit()
                    h = nvmlDeviceGetHandleByIndex(0)
                    info_before = nvmlDeviceGetMemoryInfo(h)
                    # EMPTY CACHED MEMORY
                    torch.cuda.empty_cache()
                    info_after = nvmlDeviceGetMemoryInfo(h)
                    for k, v in {'total MB': info_before.total / 1024 ** 2,
                                 'free MB': info_before.free / 1024 ** 2,
                                 'used MB': info_before.used / 1024 ** 2}.items():
                        self.writer.add_scalar('gpu memory usage/' + k, v, iteration_num)

                    for k, v in {'total MB': info_before.total / 1024 ** 2,
                                 'free after emptying MB': info_after.used / 1024 ** 2,
                                 'used after emptying MB': info_after.used / 1024 ** 2}.items():
                        self.writer.add_scalar('gpu memory usage (emptied cache)/' + k, v, iteration_num)

                # how many batches we had overall - used for logging
                iteration_num += 1
                time_prepare_training_batch_start = time.time()

            if self.main_process:
                # self.writer.add_scalar('training loss last batch', loss, epoch)
                # compute AVG losses
                for k, v in losses_epoch.items():
                    losses_epoch[k] = v / len(self.train_dataloader)

                # self.writer.add_scalars(f'Losses/train', losses_epoch, iteration_num)
                for k, v in losses_epoch.items():
                    if not k[:11] == "mask_scores":
                        self.writer.add_scalar('train/' + k, v, iteration_num)
                    else:
                        self.writer.add_scalar('train_mask_scores/' + k[12:], v, iteration_num)

                self.writer.add_scalar('Epoch', epoch, iteration_num)
                print('EPOCH AVG:', losses_epoch)

    def save_checkpoint(self, epoch, training_time, iteration_num):
        path = self.cfg.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(training_time), training_time])
        if not os.path.exists(path):
            save_dict = { 
                        'training_time': training_time,'epoch': epoch, 'iteration_num': iteration_num,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }
            torch.save(save_dict, path)

    def load_checkpoint(self, load_idx=-1, checkpoint=None):
        time_start = time.time()
        checkpoints = glob(self.cfg.checkpoint_path+'/*')

        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.cfg.checkpoint_path))
            return 0, 0, 0
        if checkpoint is None:
            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)
            path = self.cfg.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(checkpoints[load_idx]), checkpoints[load_idx]])
        else:
            path = self.cfg.checkpoint_path + '{}.tar'.format(checkpoint)
        print('Loading checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        if self.cfg.apple_warmstart:
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if (k != "mlp_semantics.6.kernel" and k != "mlp_semantics.6.bias")}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            checkpoint['model_state_dict'] = model_dict
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'],
                                                  strict= not self.cfg.loose_model_loading)
        if len(missing_keys) == 0 and not self.cfg.apple_warmstart:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        iteration_num = checkpoint['iteration_num']

        self.epoch = epoch
        print(f'Loaded checkpoint in {time.time() - time_start:.3f} seconds')
        return epoch, training_time, iteration_num

    def compute_val_loss(self, num_batches=5):
        self.model.eval()

        val_losses = defaultdict(int) 
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            with torch.no_grad():
                loss_dict = self.model.compute_loss( val_batch, self.epoch)
            for k, v in loss_dict.items():
                val_losses[k] += v.item()
            print("[VAL]: Current losses: {} ".format({k: v.item() for k, v in loss_dict.items()}))
        # free memory from validation data
        del val_batch, loss_dict
        for k, v in val_losses.items():
            val_losses[k] = v / num_batches

        return val_losses

def start_train(rank, cfg, num_devices, train_dataset, val_dataset):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:33456",
        world_size=num_devices,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    model = Model(cfg, rank)
    train_dataloader = train_dataset.get_loader_multi_gpu(rank=rank, world_size=num_devices)
    trainer = Trainer(model, train_dataloader, val_dataset, cfg, rank)
    trainer.train_model(10000)

if __name__ == '__main__':
    cfg = cfg_loader.get_config()

    if cfg.dataset_name == 'scannet':
        import dataprocessing.scannet as scannet
        semantic_valid_class_ids_torch = scannet.SCANNET_SEMANTIC_VALID_CLASS_IDS_torch
        is_foreground = scannet.is_foreground
        semantic_id2idx = scannet.SCANNET_SEMANTIC_ID2IDX
        instance_id2idx = scannet.SCANNET_INSTANCE_ID2IDX

        if not cfg.train_submission:
            val_dataset = ScanNet('val', cfg)
            train_dataset = ScanNet('train', cfg)
        else:
            val_dataset = None
            train_dataset = ScanNet('train+val', cfg)
    elif cfg.dataset_name == 'arkitscenes':
        import dataprocessing.arkitscenes as arkitscenes
        val_dataset = ARKitScenes('val', cfg, subsample_rate=cfg.subsample_rate)
        train_dataset = ARKitScenes('train', cfg, subsample_rate=cfg.subsample_rate)
        semantic_valid_class_ids_torch = arkitscenes.ARKITSCENES_SEMANTIC_VALID_CLASS_IDS_torch
        semantic_id2idx = arkitscenes.ARKITSCENES_SEMANTIC_ID2IDX
        instance_id2idx = arkitscenes.ARKITSCENES_INSTANCE_ID2IDX
        is_foreground = arkitscenes.is_foreground
    elif cfg.dataset_name == 's3dis':
        import dataprocessing.s3dis as s3dis
        val_dataset = S3DIS('val', cfg)
        train_dataset = S3DIS('train', cfg)
        semantic_valid_class_ids_torch = s3dis.S3DIS_SEMANTIC_VALID_CLASS_IDS_torch
        semantic_id2idx = s3dis.S3DIS_SEMANTIC_ID2IDX
        instance_id2idx = s3dis.S3DIS_INSTANCE_ID2IDX
        is_foreground = s3dis.is_foreground

    if cfg.fixed_seed:
        print('WARNING: fixed seed selected for training.')

    if cfg.multigpu:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn') # said to be required in pytorch docs, for num_workers > 1 in dataloader

        num_devices = torch.cuda.device_count()
        mp.spawn(start_train, nprocs=num_devices, args=(cfg, num_devices, train_dataset, val_dataset))
    else:
        model = Model(cfg, semantic_valid_class_ids_torch, semantic_id2idx, instance_id2idx, is_foreground)
        train_dataloader = train_dataset.get_loader()
        trainer = Trainer(model, train_dataloader, val_dataset, cfg)
        trainer.train_model(10000)

