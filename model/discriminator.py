# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Discriminator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import FirstBlockDiscriminator, BasicBlockDiscriminator
from self_attention import SelfAttention
from vit_encoder import ViTImageEncoder

class Discriminator(nn.Module):
  def __init__(self, dvp=3, dvpe=512, dife=512, ch=64,
               img_size=256, patch_size=16, vit_embed_dim=512, vit_depth=4,
               vit_num_heads=8, vit_mlp_ratio=4.0, vit_qk_norm=True,
               vit_init_values=0.01, vit_multiscale_layers=(5, 7, 9, 11),
               vit_pool_mode="mean", vit_num_context_views=None,
               vit_concat_pool_dim=512):
    super(Discriminator, self).__init__()

    self.dvp, self.dvpe = dvp, dvpe
    self.dife = dife
    self.ch = ch

    # input-image conditioning subnet (replaces sparams_subnet + vops_subnet)
    # separate weights from Generator's encoder -- no sharing between G and D
    self.image_encoder = ViTImageEncoder(
      img_size=img_size, patch_size=patch_size, embed_dim=vit_embed_dim,
      depth=vit_depth, num_heads=vit_num_heads, mlp_ratio=vit_mlp_ratio,
      qk_norm=vit_qk_norm, init_values=vit_init_values,
      multiscale_layers=vit_multiscale_layers,
      pool_mode=vit_pool_mode, num_context_views=vit_num_context_views,
      concat_pool_dim=vit_concat_pool_dim
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
      nn.Linear(dife + dvpe, ch * 16),
      nn.ReLU()
    )

    # image classification subnet
    self.img_subnet = nn.Sequential(
      FirstBlockDiscriminator(3, ch, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch, ch * 2, kernel_size=3,
                              stride=1, padding=1),
      # SelfAttention(ch * 2),
      BasicBlockDiscriminator(ch * 2, ch * 4, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 4, ch * 8, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 8, ch * 8, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 8, ch * 16, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 16, ch * 16, kernel_size=3,
                              stride=1, padding=1, downsample=False),
      nn.ReLU()
    )

    # output subnets
    self.out_subnet = nn.Sequential(
      nn.Linear(ch * 16, 1)
    )

  def forward(self, input_image, plucker, vp, x):
    img_feat = self.image_encoder(input_image, plucker)
    img_feat = self.image_proj_subnet(img_feat)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((img_feat, vp), 1)
    mp = self.mparams_subnet(mp)

    x = self.img_subnet(x)
    x = torch.sum(x, (2, 3))

    out = self.out_subnet(x)
    out += torch.sum(mp * x, 1, keepdim=True)

    return out
