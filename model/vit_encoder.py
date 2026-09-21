# Minimal, standalone Vision Transformer image encoder, vendored and
# simplified from TokenGS's tokengs/models/attention.py. Stripped of
# flex-attention, RoPE, and stochastic depth (DropPath) -- none of which
# apply to InSituNet's single-image-per-sample conditioning use case, and
# the first of which cannot even be imported under torch==1.13.1 (this
# repo's pinned version; see requirements.txt).
#
# PatchEmbed matches TokenGS's exactly, including its post-projection
# LayerNorm. TokenGS gets positional information from Plucker ray
# embeddings summed into the patch tokens (see tokengs/models/tokengs.py's
# _embed_encoder_input) rather than a learned absolute position table --
# that ray-embedding step is not yet ported here, so this encoder currently
# has no positional signal at all until it lands.

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None,
               act_layer=nn.GELU, drop=0.0, bias=True):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class Attention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True,
               attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm,
               qk_norm=True, fused_attn=None):
    super().__init__()
    assert dim % num_heads == 0, "dim should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    # torch==1.13.1 (this repo's pinned version) has no
    # F.scaled_dot_product_attention; auto-detect so this still picks the
    # fused path if ever run under a newer torch.
    self.fused_attn = _HAS_SDPA if fused_attn is None else fused_attn

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
    self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim, bias=proj_bias)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, N, C = x.shape
    qkv = (self.qkv(x)
           .reshape(B, N, 3, self.num_heads, self.head_dim)
           .permute(2, 0, 3, 1, 4))
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
      x = F.scaled_dot_product_attention(
          q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
    else:
      q = q * self.scale
      attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
      attn = self.attn_drop(attn)
      x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class LayerScale(nn.Module):
  def __init__(self, dim, init_values=1e-5, inplace=False):
    super().__init__()
    self.inplace = inplace
    self.gamma = nn.Parameter(init_values * torch.ones(dim))

  def forward(self, x):
    return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
               proj_bias=True, ffn_bias=True, qk_norm=True,
               init_values=0.01, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                          proj_bias=proj_bias, qk_norm=qk_norm,
                          norm_layer=norm_layer)
    self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    self.norm2 = norm_layer(dim)
    self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, bias=ffn_bias)
    self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

  def forward(self, x):
    x = x + self.ls1(self.attn(self.norm1(x)))
    x = x + self.ls2(self.mlp(self.norm2(x)))
    return x


def make_2tuple(x):
  return x if isinstance(x, tuple) else (x, x)


class PatchEmbed(nn.Module):
  """(B, C, H, W) -> (B, N, D). Exact port of TokenGS's PatchEmbed
  (tokengs/models/attention.py), reused for both RGB (in_chans=3) and
  Plucker ray (in_chans=6) patch embedding, matching TokenGS's own
  patch_embed/patch_plucker_embed pair."""
  def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512,
               norm_layer=None, flatten_embedding=True):
    super().__init__()
    image_hw = make_2tuple(img_size)
    patch_hw = make_2tuple(patch_size)
    grid = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])

    self.img_size = image_hw
    self.patch_size = patch_hw
    self.num_patches = grid[0] * grid[1]
    self.in_chans = in_chans
    self.embed_dim = embed_dim
    self.flatten_embedding = flatten_embedding

    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    _, _, H, W = x.shape
    patch_H, patch_W = self.patch_size
    assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
    assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
    x = self.proj(x)
    H, W = x.size(2), x.size(3)
    x = x.flatten(2).transpose(1, 2)
    x = self.norm(x)
    if not self.flatten_embedding:
      x = x.reshape(-1, H, W, self.embed_dim)
    return x


class ViTImageEncoder(nn.Module):
  """Vendored, simplified ViT encoder for multi-view image conditioning.

  patch_embed(rgb) + patch_plucker_embed(plucker) -> joint sequence over
  all views' patches -> stack of Block -> LayerNorm -> pool (mean, for
  now) -> (B, embed_dim). Matches TokenGS's tokengs.py _embed_encoder_input
  (patchify each view independently, concatenate into one joint sequence
  so patches from different views can attend to each other) + enc_dec.py's
  EncDecBackbone.encoder/encoder_norm.
  """
  def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512,
               depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
               qk_norm=True, init_values=0.01, multiscale_layers=(5, 7, 9, 11)):
    super().__init__()
    # Port of TokenGS's EncDecBackbone.use_multiscale=True path
    # (tokengs/models/enc_dec.py's _encode_features): rather than a single
    # LayerNorm on the last block's output, snapshot+LayerNorm the running
    # sequence at each index in `multiscale_layers` (without feeding those
    # snapshots back into the residual stream -- every block still only
    # ever sees the previous block's raw output) and concatenate the
    # snapshots channel-wise. embed_dim below ends up
    # embed_dim * len(multiscale_layers) accordingly.
    self.multiscale_layers = tuple(multiscale_layers)

    norm_layer_factory = nn.LayerNorm
    self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim,
                                  norm_layer=norm_layer_factory)
    self.patch_plucker_embed = PatchEmbed(img_size, patch_size, 6, embed_dim,
                                          norm_layer=norm_layer_factory)

    self.blocks = nn.ModuleList([
        Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
              qk_norm=qk_norm, init_values=init_values)
        for _ in range(depth)
    ])
    self.multiscale_norms = nn.ModuleList([
        nn.LayerNorm(embed_dim) for _ in self.multiscale_layers
    ])
    self.embed_dim = embed_dim * len(self.multiscale_layers)

  def pool(self, x):
    # (B, N, C) -> (B, C). Isolated on purpose: mean-pool for now, swap for
    # attention-pooling (learnable query cross-attending into patch tokens)
    # as a later follow-up without touching anything else in this class.
    return x.mean(dim=1)

  def forward(self, x, plucker):
    # x: (B, K, 3, H, W) -- K context views. plucker: (B, K, 6, H, W).
    B, K, C_in, H, W = x.shape
    x = x.reshape(B * K, C_in, H, W)
    plucker = plucker.reshape(B * K, plucker.shape[2], H, W)

    x = self.patch_embed(x) + self.patch_plucker_embed(plucker)  # (B*K, N, C)
    N, C = x.shape[1], x.shape[2]
    x = x.reshape(B, K * N, C)  # joint sequence over all K views' patches

    features = []
    for i, blk in enumerate(self.blocks):
      x = blk(x)
      if i in self.multiscale_layers:
        features.append(self.multiscale_norms[len(features)](x))
    x = torch.cat(features, dim=-1)  # (B, K*N, C * len(multiscale_layers))
    return self.pool(x)
