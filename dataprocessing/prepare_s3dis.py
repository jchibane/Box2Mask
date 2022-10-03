import sys
sys.path.append('.')

import numpy as np
import skimage.io as io
import open3d as o3d
import pyviz3d.visualizer as viz
import os
import glob
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import natsort

import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument("--scene_id", type=int, default=None,
                        help="Input the index of a scene to process. Default is None - process all scene")

parser.add_argument("--data_dir", type=str, default='./data/Stanford3dDataset_v1.2_Aligned_Version/',
                        help="Path to the original data")

S3DIS_SEMANTICS_COLORS = np.array (
    [(174, 199, 232),  # ceiling
    (152, 223, 138),  # floor
    (31, 119, 180),   # wall
    (255, 187, 120),  # column
    (188, 189, 34),   # beam
    (140, 86, 75),    # window
    (255, 152, 150),  # door
    (214, 39, 40),    # table
    (197, 176, 213),  # chair
    (148, 103, 189),  # bookcase
    (196, 156, 148),  # sofa
    (23, 190, 207),   # board
    (178, 76, 76),]   # clutter
)

INS_COLORS = np.array ([[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range (1000)])
    
def visualize (scene_name, pts, colors, normals, instances, semantics):
    viz_pth = 'visualize_scene_npy/' + scene_name
    os.makedirs (viz_pth, exist_ok=True)
    v = viz.Visualizer()
    sample_rate = 4
    
    normals_start = pts [::sample_rate*3]
    normals = normals[::sample_rate*3]
    pts = pts [::sample_rate]
    colors = colors [::sample_rate]
    instances = instances [::sample_rate]
    semantics = semantics [::sample_rate]
    
    instances_colors = INS_COLORS [instances]
    semantics_colors = S3DIS_SEMANTICS_COLORS [semantics]
        
    v.add_points ("points", pts , colors, point_size=15, visible=True)
    v.add_points ("instances", pts , instances_colors, point_size=15, visible=True)
    v.add_points ("semantics", pts , semantics_colors, point_size=15, visible=True)
    norm_colors = np.array ([[0,255,0] for _ in range (len (normals))])
    v.add_lines ("normals", normals_start, normals_start + normals / 15, norm_colors, visible=False)
    v.save(viz_pth, verbose=False)
    
    
ID2NAME = {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
ID2NAME = [ID2NAME [i] for i in range (13)]
NAME2ID = {}
for i in range (13):
    NAME2ID [ID2NAME [i]] = i
    
def get_labels (scene_name, scene_data, data_dir):
    area = scene_name.split ('.') [0]
    name = scene_name.split ('.') [1]
    instance_pths = glob.glob (data_dir + '/' + area + '/' + name + '/Annotations/*.txt')
    
    scene_pts = scene_data [:,:3] # Scene point cloud 
    pt_tree = KDTree (scene_pts)
    
    error = 0
    instances = np.zeros ((len (scene_data), 1), dtype=np.int32) - 1
    semantics = np.zeros ((len (scene_data), 1), dtype=np.float32) - 1
    
    # Use nearest neighbor to find corresponding point indexes in the scenes PC of instances
    for instance_id, pth in enumerate (instance_pths):
        class_name = pth.split ('/')[-1].split('_')[0]
        if not (class_name in NAME2ID.keys ()):
            if class_name == 'stairs':
                class_name = 'clutter'
        semantic_id = NAME2ID [class_name]
        # Load instance point cloud
        instance_data = np.loadtxt (pth)
        instance_pts = instance_data [:, :3]
        instance_colors = instance_data [:, 3:]
        # Find corresponding indices in the scene points
        dist, pt_indexs = pt_tree.query(instance_pts, k=1)
        instances [pt_indexs] = instance_id
        semantics [pt_indexs] = semantic_id
        error += dist.sum ()
        
    decided = (instances >= 0)[:, 0]
    
    # For some points are not annotated, use the label from nearby points
    pt_tree = KDTree (scene_pts [decided])
    dist, decided_indexs = pt_tree.query(scene_pts [~decided], k=1)
    
    instances [~decided] = instances [decided][decided_indexs]
    semantics [~decided] = semantics [decided][decided_indexs]
    
    assert (instances.min ()) >= 0
    assert (semantics.min ()) >= 0
    
    # Avoiding duplicate instances -> instance ids are contiguous from 0
    remap_id = np.array (range (instances.max () + 1))
    for new_id, old_id in enumerate (np.unique (instances)):
        remap_id [old_id] = new_id
    instances = remap_id [instances].astype (np.float32)
    unique_instances = np.unique (instances)
    
    assert np.all(unique_instances == range(len(unique_instances)))
    
    return instances, semantics

def read_scene_txt (name, data_dir):
    area = name.split ('.') [0]
    name = name.split ('.') [1]
    
    pts = np.loadtxt (os.path.join (data_dir + '/' + area, name, name + '.txt'))
    return pts

def preprocess_s3dis (data_dir, scene_id):
    scene_list = []
    for i in range (1, 7):
        area = data_dir + '/Area_' + str (i)
        tmp = glob.glob (area + '/*')
        for scene_name in tmp:
            scene_name = scene_name.split ('/')[-2] + '.' + scene_name.split ('/')[-1]
            scene_list.append (scene_name)

    scene_list = natsort.natsorted (scene_list)
    
    if scene_id is not None:
        scene_list = scene_list [scene_id:scene_id+1]
    
    for scene_name in scene_list:
        area = scene_name.split ('.') [0]
        name = scene_name.split ('.') [1]
        save_dir = 'data/s3dis/' + area + '/'
        scene_pth = os.path.join (save_dir, name + '.normals.instance.npy')
        
        os.makedirs (save_dir, exist_ok=True)
        
        scene_data = read_scene_txt (scene_name, data_dir)
        instances, semantics = get_labels (scene_name, scene_data, data_dir)
        normals = np.load (data_dir + '/normals/' + scene_name + '.npy')
        data = np.concatenate ([scene_data, normals, semantics, instances], 1)
        
        pts = data [:,:3].astype (np.float32)
        colors = data [:,3:6].astype (np.float32)
        normals = data [:,6:9].astype (np.float32)
        semantics = data [:, -2].astype (np.int32)
        instances = data [:, -1].astype (np.int32)
        
        # visualize (scene_name, pts - pts.mean (0), colors, normals, instances, semantics)
        np.save (scene_pth, data)
        print ("saved ", scene_pth)
        
cfg = parser.parse_args()
preprocess_s3dis (cfg.data_dir, cfg.scene_id)     