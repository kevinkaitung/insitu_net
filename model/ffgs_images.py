import json
import os
import random

import numpy as np
from skimage import io

import torch
from torch.utils.data import Dataset


def _discover_scenes(root):
  scenes = []
  for name in sorted(os.listdir(root)):
    transforms_path = os.path.join(root, name, "gaussian_splat", "transforms.json")
    if os.path.isfile(transforms_path):
      scenes.append(name)
  return scenes


def _split_scenes(scenes, test_scene_fraction, scene_split_seed):
  scenes = sorted(scenes)
  shuffled = scenes[:]
  random.Random(scene_split_seed).shuffle(shuffled)
  n_test = max(1, round(test_scene_fraction * len(shuffled)))
  test_scene_set = set(shuffled[:n_test])
  train_scenes = [s for s in scenes if s not in test_scene_set]
  test_scenes = [s for s in scenes if s in test_scene_set]
  return train_scenes, test_scenes


def _scene_center(frames):
  # Least-squares point closest to every camera's optical axis. Ported from
  # baselines/ViSNeRF/prepare_cq500.py's lookat_and_radius() (already
  # validated against this exact dataset) -- more robust than averaging
  # camera positions, since it doesn't assume symmetric sphere coverage.
  # transform_matrix is c2w OpenGL convention: translation = col 3, forward = -col 2.
  T = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
  o = T[:, :3, 3]
  d = -T[:, :3, 2]
  d /= np.linalg.norm(d, axis=1, keepdims=True)
  A = np.zeros((3, 3))
  b = np.zeros(3)
  for oi, di in zip(o, d):
    P = np.eye(3) - np.outer(di, di)
    A += P
    b += P @ oi
  return np.linalg.solve(A, b)


def _load_transforms(root, scene):
  transforms_path = os.path.join(root, scene, "gaussian_splat", "transforms.json")
  with open(transforms_path) as f:
    return json.load(f)


def _count_views(root, scene):
  # Only reads the frame count and discards the rest -- called once per
  # scene at dataset construction, just to build the flat (scene, view_idx)
  # index below. The full transforms.json is re-read in __getitem__.
  return len(_load_transforms(root, scene)["frames"])


def _load_scene(root, scene):
  # Re-parses transforms.json and re-derives positions/center from scratch
  # on every call -- mirrors methods/TokenGS's dl3dv.py (DL3DV10K.get_data),
  # which re-opens and re-parses a scene's transforms.json on every access
  # rather than caching it. Nothing about a scene is retained between calls,
  # so dataset memory stays flat regardless of how many scenes are in the
  # split (important once root holds tens of thousands of scenes, and
  # doubly so since DataLoader(num_workers>0) forks the whole Dataset object
  # into every worker process).
  meta = _load_transforms(root, scene)
  file_paths = [os.path.join(root, scene, "gaussian_splat", fr["file_path"])
                for fr in meta["frames"]]
  positions = np.array([fr["transform_matrix"] for fr in meta["frames"]])[:, :3, 3]
  center = _scene_center(meta["frames"])
  return file_paths, positions, center


class FFGSImageDataset(Dataset):
  """Loads (input_image, vparams) -> image samples from a directory of
  per-scene volume-rendered views (see
  datasets/rendered_images/CQ500_processed_new).

  Each scene subfolder under `root` is expected to contain:
    <scene>/gaussian_splat/transforms.json  (frames: file_path, transform_matrix)
    <scene>/gaussian_splat/images/*.jpg

  Camera positions are assumed to lie on a sphere centered on the scene's
  volume, always looking at that center -- verified against this dataset's
  actual transform_matrix values. The scene center is estimated via
  least-squares intersection of each camera's optical axis (ported from
  baselines/ViSNeRF/prepare_cq500.py's lookat_and_radius()), which recovers
  the true center to ~1e-6 absolute precision on the sample data and needs
  no external metadata or assumption of symmetric camera coverage.

  Loads lazily from disk, following methods/TokenGS's tokengs/data/static/
  dl3dv.py (DL3DV10K): __init__ only discovers scene paths and each scene's
  view count (one cheap transforms.json parse per scene, nothing retained
  beyond the count); every other per-scene value -- positions, the
  least-squares center, image bytes -- is recomputed fresh in __getitem__
  on every access, same as dl3dv.py's get_data(). Scales to root directories
  with tens of thousands of scenes without holding every scene's metadata
  in memory (and without that memory being multiplied across
  DataLoader(num_workers>0) worker processes, since each worker forks a
  copy of the whole Dataset object).
  """
  def __init__(self, root, train=True, test_scene_fraction=0.1,
               scene_split_seed=42, transform=None):
    self.root = root
    self.train = train
    self.transform = transform

    all_scenes = _discover_scenes(root)
    train_scenes, test_scenes = _split_scenes(
        all_scenes, test_scene_fraction, scene_split_seed)
    self.scenes = train_scenes if train else test_scenes

    self._index = []
    for scene in self.scenes:
      n_views = _count_views(root, scene)
      self._index.extend((scene, i) for i in range(n_views))

  def __len__(self):
    return len(self._index)

  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.item()
    scene, target_idx = self._index[index]
    file_paths, positions, center = _load_scene(self.root, scene)
    num_views = len(file_paths)

    if self.train:
      input_idx = random.choice([j for j in range(num_views) if j != target_idx])
    else:
      input_idx = (target_idx + num_views // 2) % num_views

    image = io.imread(file_paths[target_idx])[:, :, 0:3]
    input_image = io.imread(file_paths[input_idx])[:, :, 0:3]
    rel = positions[target_idx] - center
    vparams = (rel / np.linalg.norm(rel)).astype(np.float32)

    sample = {"image": image, "input_image": input_image, "vparams": vparams}
    if self.transform:
      sample = self.transform(sample)
    return sample


class Normalize(object):
  def __call__(self, sample):
    image = (sample["image"].astype(np.float32) - 127.5) / 127.5
    input_image = (sample["input_image"].astype(np.float32) - 127.5) / 127.5
    return {"image": image, "input_image": input_image, "vparams": sample["vparams"]}


class ToTensor(object):
  def __call__(self, sample):
    image = sample["image"].transpose((2, 0, 1))
    input_image = sample["input_image"].transpose((2, 0, 1))
    return {
        "image": torch.from_numpy(image),
        "input_image": torch.from_numpy(input_image),
        "vparams": torch.from_numpy(sample["vparams"]),
    }
