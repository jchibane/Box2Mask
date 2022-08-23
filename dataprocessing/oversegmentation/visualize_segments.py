"""Visualizes instance segmentations (from scannet format)."""

import os
import json
import random
import pyviz3d.visualizer as viz
import open3d as o3d
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('path_scenes', '../../data/scannet/scans_test/', help='Path to scene .ply')
flags.DEFINE_string('path_segments', '../../data/scannet/scans_test_segmented/', help='Path to scene .seg.json')
flags.DEFINE_string('path_viewer', '../../viewer/', 'Path to the visualizations.')


def visualize_scene(scene_name):
    """Propagates the actual per-point predictions to the segments."""

    path_ply = os.path.join(FLAGS.path_scenes, f'{scene_name}/{scene_name}_vh_clean_2.ply')
    path_segs_json = os.path.join(FLAGS.path_segments, f'{scene_name}_vh_clean_2.0.010000.segs.json')
    path_viewer = os.path.join(FLAGS.path_viewer, scene_name)

    # Read ply
    mesh = o3d.io.read_triangle_mesh(path_ply)
    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    vertices_positions = np.asarray(mesh.vertices)
    vertices_positions -= np.mean(vertices_positions, axis=0)
    vertices_normals = np.asarray(mesh.vertex_normals)
    vertices_colors = np.asarray(mesh.vertex_colors)

    # Read segments from json
    with open(path_segs_json) as f:
        data = json.load(f)
    segment_indices_list = data["segIndices"]
    segment_indices_int_array = np.asarray(segment_indices_list, dtype='int32')

    # Create segment colors
    segment_colors = np.ones_like(vertices_positions)
    for segment_id in set(segment_indices_list):
        mask = segment_id == segment_indices_int_array  # point ids of segment
        segment_colors[mask] = np.array([random.random()*255, random.random()*255, random.random()*255])

    v = viz.Visualizer()
    v.add_points(scene_name+'_color', vertices_positions, vertices_colors*255, vertices_normals, point_size=25)
    v.add_points(scene_name+'_segments', vertices_positions, segment_colors, vertices_normals, point_size=25)
    v.save(path_viewer, verbose=True)


def main(_):
    scene_names = sorted([s.split('.')[0] for s in os.listdir(f'{FLAGS.path_scenes}')])
    for scene_name in scene_names:
        visualize_scene(scene_name)


if __name__ == '__main__':
    app.run(main)
