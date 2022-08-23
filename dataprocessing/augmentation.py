import numpy as np
import open3d as o3d
import random
import scipy
import matplotlib
import albumentations as A

SCANNET_ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

# mix 3d color augmentation and normalization
mix3d_albumentations_aug = A.load('dataprocessing/mix3d_albumentations_aug.yaml', data_format="yaml")
color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
# input colors should be in 0,..,255 because
# Normalize method applies: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
color_norm = A.Normalize(mean=color_mean, std=color_std)

# HUE aug
hue_aug = A.Compose([
    A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=60, val_shift_limit=50, p=1),
], p=1)

def rotate_mesh (mesh, max_xy_angle=np.pi / 100, individual_prob = 1):
    """ Randomly rotate the point clouds around z-axis (max 360 degree), x-axis and y-axis (max max_xy_angle degree)
    """
    random_z_angle = 0
    random_x_angle = 0
    random_y_angle = 0
    if random.random() < individual_prob:
      random_z_angle = np.random.uniform (0, 2*np.pi)
    if random.random() < individual_prob:
      random_x_angle = np.random.uniform (-max_xy_angle, max_xy_angle)
    if random.random() < individual_prob:
      random_y_angle = np.random.uniform (-max_xy_angle, max_xy_angle)
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((random_x_angle,random_y_angle,random_z_angle)))


def rotate_mesh_90_degree(mesh):
  """ Randomly rotate the point clouds around z-axis (random angle in 0,90,180,270 degree)
  """
  random_z_angle = [0, 0.5* np.pi, np.pi, 1.5 * np.pi][np.random.randint(0,4)]
  random_x_angle = 0
  random_y_angle = 0
  mesh.rotate(mesh.get_rotation_matrix_from_xyz((random_x_angle, random_y_angle, random_z_angle)))

def scale_mesh (mesh, min_scale=0.9, max_scale=1.1):
    """ Randomly scale the point cloud with a random scale value between min and max
    """
    scale = np.random.uniform (min_scale, max_scale)
    mesh.scale(scale, center=(0, 0, 0))

def color_jittering (colors, min=-0.05, max=0.05):
    """ Randomly jitter color 
        Input:
          Nx3 array, original point colors
        Return:
          Nx3 array, jittered point colors
    """
    jitters = np.random.uniform (min, max, colors.shape)
    jittered_colors = np.clip(jitters + colors, 0, 1)
    return jittered_colors

def random_brightness (colors, brightness_limit=0.2):
  brighness_aug = A.RandomBrightnessContrast(p=1.0, brightness_limit=brightness_limit, contrast_limit=0.0, always_apply=True)
  colors = brighness_aug (image=colors.astype (np.float32)) ["image"]
  return colors 

def elastic_distortion( coords, granularity, magnitude):
  """Apply elastic distortion on sparse coordinate space.
    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
  blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
  blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
  blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
  coords_min = coords.min(0)

  # Create Gaussian noise tensor of the size given by granularity.
  noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
  noise = np.random.randn(*noise_dim, 3).astype(np.float32)

  # Smoothing.
  for _ in range(2):
    noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

  # Trilinear interpolate noise filters for each spatial dimensions.
  ax = [
      np.linspace(d_min, d_max, d)
      for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)
  ]
  interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
  coords += interp(coords) * magnitude
  return coords


class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,1] or a PIL image"""

  def __init__(self, trans_range_ratio=0.1):
    """
    trans_range_ratio: ratio of translation i.e. 1.0 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio

  def __call__(self, feats):
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 1.0 * 2 * self.trans_range_ratio
      feats[:, :3] = np.clip(tr + feats[:, :3], 0, 1)
    return feats
  
class RandomBrightness (object):
  """Randomly modify the brightness of the image"""
  def __init__ (self, factor_range=0.2):
    self.factor_range = factor_range
  
  def __call__ (self, feats):
    hsv = matplotlib.colors.rgb_to_hsv (feats)
    factor_range = self.factor_range
    factor = np.random.uniform (1 - factor_range, 1 + factor_range) 
    hsv [:,2] *= factor
    hsv = np.clip (hsv, 0, 1)
    rgb = matplotlib.colors.hsv_to_rgb (feats)
    return rgb

class ChromaticAutoContrast(object):

  def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
    self.randomize_blend_factor = randomize_blend_factor
    self.blend_factor = blend_factor

  def __call__(self, feats):
    if random.random() < 1.0:
      lo = feats[:, :3].min(0, keepdims=True)
      hi = feats[:, :3].max(0, keepdims=True)
      assert hi.max() <= 1, f"invalid color value. Color is supposed to be [0-1]"

      scale = 1.0 / (hi - lo)

      contrast_feats = (feats[:, :3] - lo) * scale

      blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
      feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
    return feats

def apply_mix3d_color_aug(color):
  color = color * 255 # needs to be in [0,255]
  pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
  color = np.squeeze(mix3d_albumentations_aug(image=pseudo_image)["image"])

  # normalize color information
  pseudo_image = color[np.newaxis, :, :]
  color = np.squeeze(color_norm(image=pseudo_image)["image"])
  return color

def apply_hue_aug(color):
    color = color * 255  # needs to be in [0,255]
    pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
    pseudo_image = hue_aug(image=pseudo_image)["image"]
    pseudo_image = mix3d_albumentations_aug(image=pseudo_image)["image"]
    color = np.squeeze(pseudo_image)

    # normalize color information
    pseudo_image = color[np.newaxis, :, :]
    color = np.squeeze(color_norm(image=pseudo_image)["image"])
    return color

# Elastic distortion implemented like in HAIS
def HAIS_elastic( x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(x).max(0).astype(np.int32)//int(gran) + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x + g(x) * mag