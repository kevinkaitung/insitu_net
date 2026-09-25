# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlockGenerator
from self_attention import SelfAttention
from vit_encoder import ViTImageEncoder

class Generator(nn.Module):
  def __init__(self, dvp=3, dvpe=512, dife=512, ch=64,
               img_size=256, patch_size=16, vit_embed_dim=512, vit_depth=4,
               vit_num_heads=8, vit_mlp_ratio=4.0, vit_qk_norm=True,
               vit_init_values=0.01, vit_multiscale_layers=(5, 7, 9, 11),
               vit_pool_mode="mean", vit_num_context_views=None,
               vit_concat_pool_dim=512, vit_token_proj_dim=4, vit_token_mix_dim=1):
    super(Generator, self).__init__()

    self.dvp, self.dvpe = dvp, dvpe
    self.dife = dife
    self.ch = ch

    # input-image conditioning subnet (replaces sparams_subnet + vops_subnet)
    self.image_encoder = ViTImageEncoder(
      img_size=img_size, patch_size=patch_size, embed_dim=vit_embed_dim,
      depth=vit_depth, num_heads=vit_num_heads, mlp_ratio=vit_mlp_ratio,
      qk_norm=vit_qk_norm, init_values=vit_init_values,
      multiscale_layers=vit_multiscale_layers,
      pool_mode=vit_pool_mode, num_context_views=vit_num_context_views,
      concat_pool_dim=vit_concat_pool_dim, token_proj_dim=vit_token_proj_dim,
      token_mix_dim=vit_token_mix_dim
    )
    self.image_proj_subnet = nn.Sequential(
      nn.Linear(self.image_encoder.embed_dim, dife), nn.ReLU(),
      nn.Linear(dife, dife), nn.ReLU()
    )

    # view parameters subnet
    self.vparams_subnet = nn.Sequential(
      nn.Linear(dvp, dvpe), nn.ReLU(),
      nn.Linear(dvpe, dvpe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dife + dvpe, ch * 16 * 4 * 4, bias=False)
    )

    # image generation subnet
    self.img_subnet = nn.Sequential(
      BasicBlockGenerator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 16, ch * 8, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1),
      # SelfAttention(ch * 2),
      BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(ch),
      nn.ReLU(),
      nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1),
      nn.Tanh()
    )

  def forward(self, input_image, plucker, vp):
    img_feat = self.image_encoder(input_image, plucker)
    img_feat = self.image_proj_subnet(img_feat)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((img_feat, vp), 1)
    mp = self.mparams_subnet(mp)

    x = mp.view(mp.size(0), self.ch * 16, 4, 4)
    x = self.img_subnet(x)

    return x
