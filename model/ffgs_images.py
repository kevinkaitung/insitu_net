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


def _to_opencv_convention(c2w):
  # Exact port of methods/TokenGS/tokengs/data/static/dl3dv.py's
  # DL3DV10K.load_cameras (lines 111-120). Confirmed by direct usage
  # (TokenGS trained on this exact CQ500 dataset with this exact
  # conversion, unmodified) and re-verified numerically here: the row
  # operations (negate row 2, swap rows 0/1) are a *world*-axis remap, not
  # just a per-camera local-frame fix -- so this must be applied to every
  # frame's raw transform_matrix, consistently, before any further
  # geometry (recentering, ray computation) is done with it.
  c2w = c2w.copy()
  c2w[2, :] *= -1
  c2w = c2w[[1, 0, 2, 3], :]
  c2w[0:3, 1:3] *= -1
  return c2w


def _recenter_c2w(c2w_ref, c2w):
  # TokenGS's first_cam normalization (tokengs/data/provider.py:181):
  #   c2ws = torch.inverse(c2ws[0]).unsqueeze(0) @ c2ws
  # Expresses `c2w` relative to `c2w_ref` -- c2w_ref itself becomes the
  # identity. With multiple context views, c2w_ref is the first context
  # view; only that one recenters to identity, the rest do not.
  return np.linalg.inv(c2w_ref) @ c2w


def _plucker_embedding(c2w, fx, fy, cx, cy, H, W):
  # Numpy port of methods/TokenGS/tokengs/utils/data.py's
  # get_rays_from_uvs + ray_condition (lines 126-158). c2w is expected
  # already converted (_to_opencv_convention) and recentered
  # (_recenter_c2w). Returns (6, H, W): moment (o x d) then direction d,
  # matching TokenGS's channel order.
  i, j = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5,
                     np.arange(H, dtype=np.float64) + 0.5)
  dirs = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)
  dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

  rays_d = dirs @ c2w[:3, :3].T
  rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
  moment = np.cross(rays_o, rays_d)

  plucker = np.concatenate([moment, rays_d], axis=-1)  # (H, W, 6)
  return plucker.transpose(2, 0, 1).astype(np.float32)  # (6, H, W)


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
  # Re-parses transforms.json and re-derives everything from scratch on
  # every call -- mirrors methods/TokenGS's dl3dv.py (DL3DV10K.get_data),
  # which re-opens and re-parses a scene's transforms.json on every access
  # rather than caching it. Nothing about a scene is retained between
  # calls, so dataset memory stays flat regardless of how many scenes are
  # in the split (important once root holds tens of thousands of scenes,
  # and doubly so since DataLoader(num_workers>0) forks the whole Dataset
  # object into every worker process).
  #
  # Camera axis-convention conversion happens here, once per scene load
  # (mirroring DL3DV10K.load_cameras, which is called once per scene too) --
  # everything downstream (recentering, scaling, ray computation) operates
  # on already-converted matrices.
  meta = _load_transforms(root, scene)
  file_paths = [os.path.join(root, scene, "gaussian_splat", fr["file_path"])
                for fr in meta["frames"]]
  transform_matrices = np.stack([
      _to_opencv_convention(np.array(fr["transform_matrix"], dtype=np.float64))
      for fr in meta["frames"]
  ])
  intrinsics = (meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"], meta["w"], meta["h"])
  return file_paths, transform_matrices, intrinsics


class FFGSImageDataset(Dataset):
  """Loads (input_images, vparams) -> image samples from a directory of
  per-scene volume-rendered views (see
  datasets/rendered_images/CQ500_processed_new).

  Each scene subfolder under `root` is expected to contain:
    <scene>/gaussian_splat/transforms.json  (frames: file_path, transform_matrix)
    <scene>/gaussian_splat/images/*.jpg

  Each sample has `num_context_views` context images (the model's input)
  and one target image (what the model predicts). Camera poses are
  converted to OpenCV convention (_to_opencv_convention, matching
  TokenGS's DL3DV10K.load_cameras exactly) and recentered relative to the
  first context view (_recenter_c2w, TokenGS's first_cam normalization),
  so `vparams` (the target's recentered translation) means "where the
  target is, relative to wherever the first context view was taken from."
  Translations are scaled by a fixed, user-provided `scene_scale` (no
  per-scene auto-computation -- matches TokenGS's camera_scale_method=
  'constant', which is what it uses for most presets).

  Loads lazily from disk, following methods/TokenGS's tokengs/data/static/
  dl3dv.py (DL3DV10K): __init__ only discovers scene paths and each
  scene's view count (one cheap transforms.json parse per scene, nothing
  retained beyond the count); every other per-scene value is recomputed
  fresh in __getitem__ on every access, same as dl3dv.py's get_data().
  """
  def __init__(self, root, train=True, test_scene_fraction=0.1,
               scene_split_seed=42, num_context_views=6, scene_scale=1.0,
               transform=None):
    self.root = root
    self.train = train
    self.num_context_views = num_context_views
    self.scene_scale = scene_scale
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
    file_paths, transform_matrices, (fx, fy, cx, cy, w, h) = _load_scene(self.root, scene)
    num_views = len(file_paths)
    K = self.num_context_views

    candidates = [j for j in range(num_views) if j != target_idx]
    if self.train:
      input_idx = random.sample(candidates, K)
    else:
      # Placeholder deterministic rule -- distinct from the manual,
      # per-scene context-view selection discussed for eval (mirroring
      # TokenGS's evaluation_idx_*.json), which is a separate follow-up.
      input_idx = [(target_idx + num_views // 2 + k) % num_views for k in range(K)]

    image = io.imread(file_paths[target_idx])[:, :, 0:3]
    input_image = np.stack([io.imread(file_paths[i])[:, :, 0:3] for i in input_idx])

    c2ws_context = transform_matrices[input_idx].copy()
    c2ws_context[:, :3, 3] *= self.scene_scale
    c2w_target = transform_matrices[target_idx].copy()
    c2w_target[:3, 3] *= self.scene_scale

    c2w_ref = c2ws_context[0]  # first context view -- the recentering reference
    c2ws_context_rel = np.stack([_recenter_c2w(c2w_ref, c) for c in c2ws_context])
    c2w_target_rel = _recenter_c2w(c2w_ref, c2w_target)

    vparams = c2w_target_rel[:3, 3].astype(np.float32)
    plucker = np.stack([_plucker_embedding(c, fx, fy, cx, cy, h, w)
                        for c in c2ws_context_rel])  # (K, 6, H, W)

    sample = {"image": image, "input_image": input_image,
              "vparams": vparams, "plucker": plucker}
    if self.transform:
      sample = self.transform(sample)
    return sample


class Normalize(object):
  def __call__(self, sample):
    image = (sample["image"].astype(np.float32) - 127.5) / 127.5
    input_image = (sample["input_image"].astype(np.float32) - 127.5) / 127.5
    return {"image": image, "input_image": input_image,
            "vparams": sample["vparams"], "plucker": sample["plucker"]}


class ToTensor(object):
  def __call__(self, sample):
    image = sample["image"].transpose((2, 0, 1))            # (3, H, W)
    input_image = sample["input_image"].transpose((0, 3, 1, 2))  # (K, 3, H, W)
    return {
        "image": torch.from_numpy(image),
        "input_image": torch.from_numpy(input_image),
        "vparams": torch.from_numpy(sample["vparams"]),
        "plucker": torch.from_numpy(sample["plucker"]),      # already (K, 6, H, W)
    }
