# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# main file for training

from __future__ import absolute_import, division, print_function

import os
import argparse
import math
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid

from torchvision import transforms
from ffgs_images import FFGSImageDataset, Normalize, ToTensor

from generator import Generator
from discriminator import Discriminator
from vgg19 import VGG19
from load_tokengs_encoder import infer_encoder_hparams, load_pretrained_tokengs_encoder
from safetensors import safe_open

# parse arguments
def parse_args():
  parser = argparse.ArgumentParser(description="InSituNet")

  parser.add_argument("--seed", type=int, default=1,
                      help="random seed (default: 1)")

  parser.add_argument("--root", required=True, type=str,
                      help="root directory containing one subfolder per "
                           "scene, each with gaussian_splat/transforms.json "
                           "and gaussian_splat/images/")
  parser.add_argument("--output-dir", required=True, type=str,
                      help="directory to save this run's checkpoints, "
                           "generated images, and wandb run files")
  parser.add_argument("--resume", type=str, default="",
                      help="path to the latest checkpoint (default: none)")

  parser.add_argument("--test-scene-fraction", type=float, default=0.1,
                      help="fraction of scenes held out entirely for "
                           "testing (default: 0.1)")
  parser.add_argument("--scene-split-seed", type=int, default=42,
                      help="random seed for the train/test scene split, "
                           "independent of --seed (default: 42)")
  parser.add_argument("--num-context-views", type=int, default=6,
                      help="number of context/input views per sample (default: 6)")
  parser.add_argument("--scene-scale", type=float, default=1.0,
                      help="fixed scale factor applied to camera translations "
                           "(TokenGS's camera_scale_method='constant'; default: 1.0)")
  parser.add_argument("--min-view-gap", type=int, default=10,
                      help="minimum window size (in view-index steps) that "
                           "context+target views are sampled from, ported from "
                           "TokenGS's Provider._get_indices_static (default: 10)")
  parser.add_argument("--max-view-gap", type=int, default=25,
                      help="maximum window size (in view-index steps) that "
                           "context+target views are sampled from, ported from "
                           "TokenGS's Provider._get_indices_static (default: 25)")

  parser.add_argument("--dvp", type=int, default=3,
                      help="dimensions of the view parameters (default: 3)")
  parser.add_argument("--dvpe", type=int, default=512,
                      help="dimensions of the view parameters' encode (default: 512)")
  parser.add_argument("--dife", type=int, default=512,
                      help="dimensions of the input image feature encode (default: 512)")
  parser.add_argument("--ch", type=int, default=64,
                      help="channel multiplier (default: 64)")

  parser.add_argument("--vit-img-size", type=int, default=256,
                      help="input image resolution for the ViT image encoder (default: 256)")
  parser.add_argument("--vit-patch-size", type=int, default=16,
                      help="patch size for the ViT image encoder (default: 16)")
  parser.add_argument("--vit-embed-dim", type=int, default=512,
                      help="embedding dim of the ViT image encoder (default: 512)")
  parser.add_argument("--vit-depth", type=int, default=12,
                      help="number of transformer blocks in the ViT image encoder (default: 4)")
  parser.add_argument("--vit-num-heads", type=int, default=8,
                      help="number of attention heads in the ViT image encoder (default: 8)")
  parser.add_argument("--vit-mlp-ratio", type=float, default=4.0,
                      help="MLP hidden-dim ratio in the ViT image encoder (default: 4.0)")
  parser.add_argument("--no-vit-qk-norm", action="store_true", default=False,
                      help="disable QK-norm in the ViT image encoder")
  parser.add_argument("--vit-init-values", type=float, default=0.01,
                      help="LayerScale init value in the ViT image encoder; "
                           "pass 0 to disable LayerScale (default: 0.01)")
  parser.add_argument("--vit-use-multiscale", action="store_true", default=True,
                      help="use TokenGS's multiscale encoder (concatenate "
                           "LayerNorm'd snapshots at --vit-multiscale-layers "
                           "instead of a single final norm); default: True, "
                           "matching every known TokenGS config")
  parser.add_argument("--no-vit-use-multiscale", dest="vit_use_multiscale",
                      action="store_false",
                      help="disable the multiscale encoder -- falls back to "
                           "a single LayerNorm snapshot at the last block")
  parser.add_argument("--vit-multiscale-layers", type=int, nargs="+", default=[5, 7, 9, 11],
                      help="block indices to snapshot+concatenate when "
                           "--vit-use-multiscale is set (default: 5 7 9 11, "
                           "TokenGS's own default; ignored otherwise)")
  parser.add_argument("--vit-pool-mode", type=str, default="mean",
                      choices=["mean", "concat", "proj_and_concat"],
                      help="how the ViT encoder's per-token features collapse "
                           "into one vector: 'mean' averages over every patch "
                           "token (parameter-free, works for any number of "
                           "context views); 'concat' flattens every patch "
                           "token across all views and Linear-projects the "
                           "result to --vit-concat-pool-dim -- this fixes the "
                           "encoder to exactly --num-context-views views, and "
                           "the projection layer's parameter count scales as "
                           "num_context_views * patches_per_view * embed_dim * "
                           "len(multiscale_layers), which gets very large very "
                           "fast; 'proj_and_concat' first applies one shared "
                           "Linear (same weights for every patch token, à la "
                           "PatchEmbed/PointNet) down to --vit-token-proj-dim, "
                           "then flattens -- also fixes the encoder to exactly "
                           "--num-context-views views, but the flatten is on "
                           "num_context_views * patches_per_view * "
                           "vit_token_proj_dim, avoiding 'concat's blowup "
                           "(default: mean)")
  parser.add_argument("--vit-concat-pool-dim", type=int, default=512,
                      help="output dim of the concat-pooling projection "
                           "Linear layer, i.e. what the rest of InSituNet "
                           "consumes as the image feature; only used when "
                           "--vit-pool-mode=concat (default: 512)")
  parser.add_argument("--vit-token-proj-dim", type=int, default=4,
                      help="output dim of the shared per-token projection "
                           "Linear layer when --vit-pool-mode=proj_and_concat "
                           "(the flattened/concatenated feature InSituNet "
                           "consumes ends up num_context_views * "
                           "patches_per_view * this; default: 4)")

  parser.add_argument("--tokengs-checkpoint", type=str, default="",
                      help="path to a pretrained TokenGS safetensors checkpoint "
                           "to initialize the ViT image encoder (patch_embed, "
                           "patch_plucker_embed, blocks, multiscale_norms) from. "
                           "--vit-patch-size/--vit-embed-dim/--vit-depth/"
                           "--vit-num-heads/--vit-mlp-ratio are inferred from "
                           "the checkpoint itself and override the corresponding "
                           "flags above. If not given, the ViT encoder trains "
                           "from scratch (default: none)")
  parser.add_argument("--freeze-vit-encoder", action="store_true", default=False,
                      help="freeze the ViT image encoder (patch_embed, "
                           "patch_plucker_embed, blocks, multiscale_norms -- "
                           "everything before pooling, which has no weights of "
                           "its own) so it's excluded from the optimizer and "
                           "never updated during training. Meant to be paired "
                           "with --tokengs-checkpoint, so TokenGS and InSituNet "
                           "are compared using the exact same encoder weights "
                           "(default: False)")

  parser.add_argument("--sn", action="store_true", default=False,
                      help="enable spectral normalization")

  parser.add_argument("--mse-loss", action="store_true", default=False,
                      help="enable mse loss")
  parser.add_argument("--perc-loss", type=str, default="relu1_2",
                      help="layer that perceptual loss is computed on (default: relu1_2)")
  parser.add_argument("--gan-loss", type=str, default="none",
                      help="gan loss (default: none)")
  parser.add_argument("--gan-loss-weight", type=float, default=0.,
                      help="weight of the gan loss (default: 0.)")

  parser.add_argument("--lr", type=float, default=1e-3,
                      help="learning rate (default: 1e-3)")
  parser.add_argument("--d-lr", type=float, default=1e-3,
                      help="learning rate of the discriminator (default: 1e-3)")
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="beta1 of Adam (default: 0.9)")
  parser.add_argument("--beta2", type=float, default=0.999,
                      help="beta2 of Adam (default: 0.999)")
  parser.add_argument("--batch-size", type=int, default=50,
                      help="batch size for training (default: 50)")
  parser.add_argument("--start-epoch", type=int, default=0,
                      help="start epoch number (default: 0)")
  parser.add_argument("--epochs", type=int, default=10,
                      help="number of epochs to train (default: 10)")

  parser.add_argument("--log-every", type=int, default=10,
                      help="log training status every given number of batches (default: 10)")
  parser.add_argument("--check-every", type=int, default=20,
                      help="save checkpoint every given number of epochs (default: 20)")
  parser.add_argument("--log-image-freq", type=int, default=100,
                      help="log a training comparison image every given number "
                           "of batches (default: 100)")

  parser.add_argument("--wandb-project", type=str, default="insitu-net",
                      help="wandb project name (default: insitu-net)")
  parser.add_argument("--wandb-run-name", type=str, default=None,
                      help="wandb run name (default: auto-generated by wandb)")
  parser.add_argument("--no-wandb", action="store_true", default=False,
                      help="disable wandb logging")

  return parser.parse_args()

# the main function
def main(args):
  accelerator = Accelerator()
  device = accelerator.device

  # unified experiment-output directory: checkpoints/ and images/ for this
  # run both live under here, separate from --root (the dataset)
  ckpt_dir = os.path.join(args.output_dir, "checkpoints")
  img_dir = os.path.join(args.output_dir, "images")
  if accelerator.is_main_process:
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

  if not args.no_wandb and accelerator.is_main_process:
    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
              config=vars(args), dir=args.output_dir)

  def wandb_log(data):
    if not args.no_wandb and accelerator.is_main_process:
      wandb.log(data)

  def save_comparison_image(path, wandb_key, epoch, context, gt, pred):
    # context/gt/pred are expected in the generator's raw [-1, 1] Tanh output
    # range, captured before any loss-specific renormalization (e.g.
    # perceptual loss's ImageNet normalization further down the training
    # loop reassigns `image`/`fake_image` in place -- callers must snapshot
    # before that). context is (B, K, 3, H, W) -- one row per context view,
    # then gt (target) / pred (generated).
    n = min(gt.size(0), 8)
    context_rows = [context[:n, k] for k in range(context.size(1))]
    comparison = torch.cat(context_rows + [
        gt[:n],
        pred.view(gt.size(0), 3, 256, 256)[:n],
    ])
    # this is used to normalize back from the generator's raw [-1, 1] Tanh output range to image RGB range [0, 1]
    comparison = ((comparison.cpu() + 1.) * .5).clamp(0, 1)
    grid = make_grid(comparison, nrow=n)
    save_image(grid, path)
    wandb_log({"epoch": epoch, wandb_key: wandb.Image(grid)})

  # log hyperparameters
  if accelerator.is_main_process:
    print(args)

  # set random seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # each DataLoader worker process needs its own random/numpy seed --
  # torch's own RNG is auto-diversified per worker, but Python's global
  # `random` module (used by FFGSImageDataset's random input-view pairing)
  # is not, so without this, workers forked from the same parent can
  # inherit identical `random` state and produce correlated "random" pairs
  def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

  # data loader
  train_dataset = FFGSImageDataset(
      root=args.root, train=True,
      test_scene_fraction=args.test_scene_fraction,
      scene_split_seed=args.scene_split_seed,
      num_context_views=args.num_context_views,
      scene_scale=args.scene_scale,
      min_view_gap=args.min_view_gap, max_view_gap=args.max_view_gap,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  test_dataset = FFGSImageDataset(
      root=args.root, train=False,
      test_scene_fraction=args.test_scene_fraction,
      scene_split_seed=args.scene_split_seed,
      num_context_views=args.num_context_views,
      scene_scale=args.scene_scale,
      min_view_gap=args.min_view_gap, max_view_gap=args.max_view_gap,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  kwargs = {"num_workers": 4, "pin_memory": True, "worker_init_fn": _worker_init_fn}
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=True, **kwargs)

  # model
  def weights_init(m):
    if isinstance(m, nn.Linear):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def add_sn(m):
    for name, c in m.named_children():
      m.add_module(name, add_sn(c))
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      return nn.utils.spectral_norm(m, eps=1e-4)
    else:
      return m

  # ViT image-encoder hyperparameters: when --tokengs-checkpoint is given,
  # --vit-patch-size/--vit-embed-dim/--vit-depth/--vit-num-heads/
  # --vit-mlp-ratio/--vit-init-values are overridden by what that checkpoint's
  # own tensor shapes actually require (see load_tokengs_encoder.py) --
  # img_size and multiscale_layers can't be recovered from a checkpoint (no
  # tensor shape depends on either), so those stay CLI-driven either way.
  vit_kwargs = dict(img_size=args.vit_img_size, patch_size=args.vit_patch_size,
                    vit_embed_dim=args.vit_embed_dim, vit_depth=args.vit_depth,
                    vit_num_heads=args.vit_num_heads, vit_mlp_ratio=args.vit_mlp_ratio,
                    vit_qk_norm=not args.no_vit_qk_norm,
                    vit_init_values=args.vit_init_values)

  if args.tokengs_checkpoint:
    with safe_open(args.tokengs_checkpoint, framework="pt") as f:
      shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    inferred = infer_encoder_hparams(shapes)
    inferred.pop("num_multiscale")
    if accelerator.is_main_process:
      print("=> inferring ViT encoder hyperparameters from --tokengs-checkpoint "
            "{}: {} (overrides --vit-patch-size/--vit-embed-dim/--vit-depth/"
            "--vit-num-heads/--vit-mlp-ratio/--vit-init-values/--no-vit-qk-norm)"
            .format(args.tokengs_checkpoint, inferred))
    vit_kwargs.update(patch_size=inferred["patch_size"],
                      vit_embed_dim=inferred["embed_dim"],
                      vit_depth=inferred["depth"],
                      vit_num_heads=inferred["num_heads"],
                      vit_mlp_ratio=inferred["mlp_ratio"],
                      vit_qk_norm=inferred["qk_norm"],
                      vit_init_values=inferred["init_values"])

  vit_multiscale_layers = (tuple(args.vit_multiscale_layers) if args.vit_use_multiscale
                           else (vit_kwargs["vit_depth"] - 1,))
  if max(vit_multiscale_layers) >= vit_kwargs["vit_depth"]:
    raise ValueError(
        "--vit-multiscale-layers {} has an index out of range for "
        "--vit-depth {} (block indices are 0..depth-1); this combination "
        "would give the ViT encoder's forward pass zero snapshots to "
        "concatenate. --vit-multiscale-layers' default (5, 7, 9, 11) needs "
        "--vit-depth >= 12 (or --tokengs-checkpoint, which infers the "
        "correct depth), or pass --no-vit-use-multiscale / matching "
        "--vit-multiscale-layers for a shallower encoder."
        .format(list(vit_multiscale_layers), vit_kwargs["vit_depth"]))
  vit_kwargs["vit_multiscale_layers"] = vit_multiscale_layers
  vit_kwargs["vit_pool_mode"] = args.vit_pool_mode
  vit_kwargs["vit_num_context_views"] = args.num_context_views
  vit_kwargs["vit_concat_pool_dim"] = args.vit_concat_pool_dim
  vit_kwargs["vit_token_proj_dim"] = args.vit_token_proj_dim

  per_token_dim = vit_kwargs["vit_embed_dim"] * len(vit_multiscale_layers)
  num_tokens = args.num_context_views * (vit_kwargs["img_size"] // vit_kwargs["patch_size"]) ** 2
  if args.vit_pool_mode == "concat":
    flatten_dim = num_tokens * per_token_dim
    n_proj_params = flatten_dim * args.vit_concat_pool_dim
    if accelerator.is_main_process:
      print("=> --vit-pool-mode=concat: concat_pool_proj is Linear({}, {}) "
            "= {:.2f}B parameters ({:.1f} GiB in fp32)"
            .format(flatten_dim, args.vit_concat_pool_dim,
                    n_proj_params / 1e9, n_proj_params * 4 / 2**30))
  elif args.vit_pool_mode == "proj_and_concat":
    n_proj_params = per_token_dim * args.vit_token_proj_dim + args.vit_token_proj_dim
    output_dim = num_tokens * args.vit_token_proj_dim
    if accelerator.is_main_process:
      print("=> --vit-pool-mode=proj_and_concat: token_proj is Linear({}, {}) "
            "= {:,} parameters (shared across all {} tokens), flattened "
            "output dim {:,}"
            .format(per_token_dim, args.vit_token_proj_dim, n_proj_params,
                    num_tokens, output_dim))

  # concat_pool_proj/token_proj (depending on --vit-pool-mode) have no
  # TokenGS counterpart -- TokenGS has no pooling layer of its own at all,
  # that's entirely InSituNet's own addition -- so they always load with
  # their own random init rather than from the checkpoint.
  if args.vit_pool_mode == "concat":
    encoder_allow_missing = ("concat_pool_proj.weight", "concat_pool_proj.bias")
  elif args.vit_pool_mode == "proj_and_concat":
    encoder_allow_missing = ("token_proj.weight", "token_proj.bias")
  else:
    encoder_allow_missing = ()

  g_model = Generator(dvp=args.dvp, dvpe=args.dvpe, dife=args.dife, ch=args.ch, **vit_kwargs)
  g_model.apply(weights_init)
  # if args.sn:
  #   g_model = add_sn(g_model)

  if args.gan_loss != "none":
    d_model = Discriminator(dvp=args.dvp, dvpe=args.dvpe, dife=args.dife, ch=args.ch, **vit_kwargs)
    d_model.apply(weights_init)
    if args.sn:
      d_model = add_sn(d_model)

  # warm-start the ViT encoder(s) from a pretrained TokenGS checkpoint --
  # after weights_init (which would otherwise overwrite this with a fresh
  # orthogonal init) and before --resume below (--resume, if also given,
  # still takes precedence: it fully overwrites both models afterward,
  # including the encoder, since it represents further-along InSituNet
  # training state rather than just a starting point)
  if args.tokengs_checkpoint:
    missing, unexpected = load_pretrained_tokengs_encoder(
        g_model.image_encoder, args.tokengs_checkpoint, vit_multiscale_layers,
        allow_missing=encoder_allow_missing)
    assert not missing and not unexpected, (missing, unexpected)
    if args.gan_loss != "none":
      missing, unexpected = load_pretrained_tokengs_encoder(
          d_model.image_encoder, args.tokengs_checkpoint, vit_multiscale_layers,
          allow_missing=encoder_allow_missing)
      assert not missing and not unexpected, (missing, unexpected)
    if accelerator.is_main_process:
      print("=> loaded pretrained TokenGS encoder from {} into g_model{}"
            .format(args.tokengs_checkpoint,
                    " and d_model" if args.gan_loss != "none" else ""))

  # Freeze the ViT image encoder (patch_embed, patch_plucker_embed, blocks,
  # multiscale_norms -- pool() has no parameters of its own, so this covers
  # everything before it) so TokenGS and InSituNet are compared using the
  # exact same, un-fine-tuned encoder weights. requires_grad=False here is
  # also what keeps these params out of the optimizer below and out of
  # DDP's gradient sync once accelerator.prepare() wraps the model.
  if args.freeze_vit_encoder:
    if not args.tokengs_checkpoint and accelerator.is_main_process:
      print("=> WARNING: --freeze-vit-encoder with no --tokengs-checkpoint "
            "freezes a randomly-initialized encoder for the entire run")
    for p in g_model.image_encoder.parameters():
      p.requires_grad = False
    if args.gan_loss != "none":
      for p in d_model.image_encoder.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
      print("=> froze g_model.image_encoder{}".format(
          " and d_model.image_encoder" if args.gan_loss != "none" else ""))

  # loss
  if args.perc_loss != "none":
    norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
    norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
    vgg = VGG19(args.perc_loss).eval().to(device)

  mse_criterion = nn.MSELoss()
  train_losses, test_losses = [], []
  d_losses, g_losses = [], []

  # optimizer -- filtered to trainable params only, so a frozen image_encoder
  # (see --freeze-vit-encoder above) is excluded rather than just carried
  # along with permanently-None gradients
  g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, g_model.parameters()),
                           lr=args.lr, betas=(args.beta1, args.beta2))
  if args.gan_loss != "none":
    d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, d_model.parameters()),
                             lr=args.d_lr, betas=(args.beta1, args.beta2))

  # load checkpoint (into the raw, pre-accelerator.prepare() model/optimizer)
  if args.resume:
    if os.path.isfile(args.resume):
      if accelerator.is_main_process:
        print("=> loading checkpoint {}".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint["epoch"]
      g_model.load_state_dict(checkpoint["g_model_state_dict"])
      g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
      if args.gan_loss != "none":
        d_model.load_state_dict(checkpoint["d_model_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        d_losses = checkpoint["d_losses"]
        g_losses = checkpoint["g_losses"]
      train_losses = checkpoint["train_losses"]
      test_losses = checkpoint["test_losses"]
      if accelerator.is_main_process:
        print("=> loaded checkpoint {} (epoch {})"
            .format(args.resume, checkpoint["epoch"]))

  # wrap model(s)/optimizer(s)/dataloaders for distributed training -- this
  # is what actually enables multi-GPU: DDP-wraps the model(s) and swaps in
  # a distributed-aware sampler on the dataloaders
  if args.gan_loss != "none":
    g_model, d_model, g_optimizer, d_optimizer, train_loader, test_loader = accelerator.prepare(
        g_model, d_model, g_optimizer, d_optimizer, train_loader, test_loader)
  else:
    g_model, g_optimizer, train_loader, test_loader = accelerator.prepare(
        g_model, g_optimizer, train_loader, test_loader)

  # main loop
  for epoch in tqdm(range(args.start_epoch, args.epochs),
                    disable=not accelerator.is_main_process):
    # training...
    g_model.train()
    if args.gan_loss != "none":
      d_model.train()
    train_loss = torch.tensor(0., device=device)
    n_train = torch.tensor(0., device=device)
    for i, sample in enumerate(train_loader):
      image = sample["image"].to(device)
      input_image = sample["input_image"].to(device)
      plucker = sample["plucker"].to(device)
      vparams = sample["vparams"].to(device)
      g_optimizer.zero_grad()
      fake_image = g_model(input_image, plucker, vparams)

      # snapshot before the perceptual-loss branch below reassigns
      # image/fake_image to ImageNet-normalized values in place
      should_log_images = (accelerator.is_main_process and
                          i % args.log_image_freq == 0)
      if should_log_images:
        vis_image = image.detach()
        vis_fake_image = fake_image.detach()
        vis_input_image = input_image.detach()

      loss = 0.

      # gan loss
      if args.gan_loss != "none":
        # update discriminator
        d_optimizer.zero_grad()
        decision = d_model(input_image, plucker, vparams, image)

        if args.gan_loss == "vanilla":
          d_loss_real = torch.mean(F.softplus(-decision))
        elif args.gan_loss == "hinge":
          d_loss_real = torch.mean(F.relu(1. - decision))

        fake_decision = d_model(input_image, plucker, vparams, fake_image.detach())

        if args.gan_loss == "vanilla":
          d_loss_fake = torch.mean(F.softplus(fake_decision))
        elif args.gan_loss == "hinge":
          d_loss_fake = torch.mean(F.relu(1. + fake_decision))

        d_loss = d_loss_real + d_loss_fake
        accelerator.backward(d_loss)

        d_optimizer.step()

        # loss of generator
        g_optimizer.zero_grad()
        fake_decision = d_model(input_image, plucker, vparams, fake_image)

        if args.gan_loss == "vanilla":
          g_loss = args.gan_loss_weight * torch.mean(F.softplus(-fake_decision))
        elif args.gan_loss == "hinge":
          g_loss = -args.gan_loss_weight * torch.mean(fake_decision)
        loss += g_loss

      # mse loss
      if args.mse_loss:
        mse_loss = mse_criterion(image, fake_image)
        loss += mse_loss

      # perceptual loss
      if args.perc_loss != "none":
        # normalize
        image = ((image + 1.) * .5 - norm_mean) / norm_std
        fake_image = ((fake_image + 1.) * .5 - norm_mean) / norm_std

        features = vgg(image)
        fake_features = vgg(fake_image)

        perc_loss = mse_criterion(features, fake_features)
        loss += perc_loss

      accelerator.backward(loss)
      g_optimizer.step()
      batch_n = torch.tensor(float(image.size(0)), device=device)
      train_loss += loss.detach() * batch_n
      n_train += batch_n

      if should_log_images:
        save_comparison_image(
            os.path.join(img_dir, "train_epoch{:04d}_iter{:04d}.png".format(epoch, i)),
            "train/comparison", epoch, vis_input_image, vis_image, vis_fake_image)

      # log training status (each process logs its own local batch value;
      # only rank 0's is actually printed/logged)
      if i % args.log_every == 0 and accelerator.is_main_process:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
          epoch, i * image.size(0), len(train_loader.dataset),
          100. * i / len(train_loader),
          loss.item()))
        wandb_log({"epoch": epoch, "train/batch_loss": loss.item()})
        if args.gan_loss != "none":
          print("DLoss: {:.6f}, GLoss: {:.6f}".format(
            d_loss.item(), g_loss.item()))
          wandb_log({"epoch": epoch, "train/d_loss": d_loss.item(),
                    "train/g_loss": g_loss.item()})
          d_losses.append(d_loss.item())
          g_losses.append(g_loss.item())
        train_losses.append(loss.item())

    # true cross-process average: gather each process's local sum before dividing
    train_loss = accelerator.gather_for_metrics(train_loss).sum()
    n_train = accelerator.gather_for_metrics(n_train).sum()
    if accelerator.is_main_process:
      avg_train_loss = (train_loss / n_train).item()
      print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_train_loss))
      wandb_log({"epoch": epoch, "train/epoch_avg_loss": avg_train_loss})

    # testing...
    g_model.eval()
    if args.gan_loss != "none":
      d_model.eval()
    test_loss = torch.tensor(0., device=device)
    n_test = torch.tensor(0., device=device)
    with torch.no_grad():
      for i, sample in enumerate(test_loader):
        image = sample["image"].to(device)
        input_image = sample["input_image"].to(device)
        plucker = sample["plucker"].to(device)
        vparams = sample["vparams"].to(device)
        fake_image = g_model(input_image, plucker, vparams)
        batch_n = torch.tensor(float(image.size(0)), device=device)
        test_loss += mse_criterion(image, fake_image).detach() * batch_n
        n_test += batch_n

        if i == 0 and accelerator.is_main_process:
          save_comparison_image(
              os.path.join(img_dir, "test_epoch_{:04d}.png".format(epoch)),
              "test/comparison", epoch, input_image, image, fake_image)

    test_loss = accelerator.gather_for_metrics(test_loss).sum()
    n_test = accelerator.gather_for_metrics(n_test).sum()
    if accelerator.is_main_process:
      avg_test_loss = (test_loss / n_test).item()
      test_losses.append(avg_test_loss)
      print("====> Epoch: {} Test set loss: {:.4f}".format(epoch, avg_test_loss))
      wandb_log({"epoch": epoch, "test/loss": avg_test_loss})

    # saving...
    if (epoch + 1) % args.check_every == 0 or epoch == args.epochs - 1:
      accelerator.wait_for_everyone()
      unwrapped_g = accelerator.unwrap_model(g_model)
      if args.gan_loss != "none":
        unwrapped_d = accelerator.unwrap_model(d_model)
      if accelerator.is_main_process:
        print("=> saving checkpoint at epoch {}".format(epoch))
        if args.gan_loss != "none":
          torch.save({"epoch": epoch + 1,
                      "g_model_state_dict": unwrapped_g.state_dict(),
                      "g_optimizer_state_dict": g_optimizer.state_dict(),
                      "d_model_state_dict": unwrapped_d.state_dict(),
                      "d_optimizer_state_dict": d_optimizer.state_dict(),
                      "d_losses": d_losses,
                      "g_losses": g_losses,
                      "train_losses": train_losses,
                      "test_losses": test_losses},
                     os.path.join(ckpt_dir, "checkpoint_epoch{:04d}.pth.tar".format(epoch)))
        else:
          torch.save({"epoch": epoch + 1,
                      "g_model_state_dict": unwrapped_g.state_dict(),
                      "g_optimizer_state_dict": g_optimizer.state_dict(),
                      "train_losses": train_losses,
                      "test_losses": test_losses},
                     os.path.join(ckpt_dir, "checkpoint_epoch{:04d}.pth.tar".format(epoch)))

        torch.save(unwrapped_g.state_dict(),
                   os.path.join(ckpt_dir, "generator_epoch{:04d}.pth".format(epoch)))

  if not args.no_wandb and accelerator.is_main_process:
    wandb.finish()

if __name__ == "__main__":
  main(parse_args())
