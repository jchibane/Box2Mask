"""Generates the segmentations of 3d scanes given as .ply using 'segmentator' (in the cpp dir).
"""

import subprocess
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('scene_path', '../../data/scannet/scans_test/', help="Path to the .ply scenes.")
flags.DEFINE_string('segments_path', '../../data/scannet/scans_test_segmented', help='Path to the generated segments.')
flags.DEFINE_string('segmentator_path', 'cpp/segmentator', help='Path to the segmentator executable.')


def segment_scene(scene_name):
    scene_path = os.path.join(FLAGS.scene_path, f'{scene_name}/{scene_name}_vh_clean_2.ply')
    command = [FLAGS.segmentator_path, scene_path, '0.01', '20', FLAGS.segments_path]
    subprocess.call(command)

def main(_):
    if not os.path.exists(FLAGS.segments_path):
        os.makedirs(FLAGS.segments_path)
    scene_names = [file.split('.')[0] for file in os.listdir(FLAGS.scene_path)]
    for scene_name in scene_names:
        segment_scene(scene_name)


if __name__ == '__main__':
    app.run(main)
