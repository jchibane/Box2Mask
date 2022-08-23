import datasets.scannet as scannet
import config_loader as cfg_loader
import os
from glob import glob

cfg = cfg_loader.get_config(['--config','src/instances_ndf/configs/l1_lr-4_relu.txt'])

scans = glob('data/scannet/scans/*')
outfolder = os.path.join('data','scannet','gt_instance_data_txt')
os.makedirs(outfolder, exist_ok = True)

for scan in scans:
    raise # method has changes
    scene, labels = scannet.process_scene(os.path.basename(scan), cfg)
    gt_format = labels['instances'] + 1000 * labels['semantics']

    with open(os.path.join(outfolder, os.path.basename(scan)) + '.txt', 'w') as f:
        for id in gt_format:
            f.write('%d\n' % id)
    break
