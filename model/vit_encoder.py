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
  all views' patches -> stack of Block -> LayerNorm -> pool -> (B, embed_dim).
  Matches TokenGS's tokengs.py _embed_encoder_input (patchify each view
  independently, concatenate into one joint sequence so patches from
  different views can attend to each other) + enc_dec.py's
  EncDecBackbone.encoder/encoder_norm.

  `pool_mode` picks how the (B, T, C) block-stack output collapses to
  (B, embed_dim): "mean" (default) averages over all T tokens, parameter-
  free, works for any T at runtime. "concat" instead flattens all T tokens'
  channels into one (T*C,) vector per batch element and Linear-projects it
  to `concat_pool_dim` -- unlike mean pooling this needs a *fixed* T, so
  `num_context_views` must be given and every forward() call must use
  exactly that many views. T*C is typically enormous (see the assert in
  __init__), so this is meant for deliberately small configs. "proj_and_concat"
  is a cheaper middle ground: a single shared Linear(C, token_proj_dim),
  applied identically and independently to every one of the T tokens (the
  same weight-tying principle as PatchEmbed's own per-patch projection --
  Deep Sets/PointNet's "shared per-element transform, then aggregate"),
  then flattening the small per-token outputs into (T*token_proj_dim,).
  Also needs a fixed T (num_context_views), but the flatten is on
  T*token_proj_dim instead of T*C, avoiding "concat"'s blowup.
  """
  def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512,
               depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
               qk_norm=True, init_values=0.01, multiscale_layers=(5, 7, 9, 11),
               pool_mode="mean", num_context_views=None, concat_pool_dim=512,
               token_proj_dim=4):
    super().__init__()
    # Port of TokenGS's EncDecBackbone.use_multiscale=True path
    # (tokengs/models/enc_dec.py's _encode_features): rather than a single
    # LayerNorm on the last block's output, snapshot+LayerNorm the running
    # sequence at each index in `multiscale_layers` (without feeding those
    # snapshots back into the residual stream -- every block still only
    # ever sees the previous block's raw output) and concatenate the
    # snapshots channel-wise. Per-token channel count below ends up
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

    self.pool_mode = pool_mode
    self.num_context_views = num_context_views
    per_token_dim = embed_dim * len(self.multiscale_layers)
    if pool_mode == "concat":
      assert num_context_views is not None, (
          "num_context_views is required when pool_mode='concat' -- unlike "
          "mean-pooling, flattening every patch token needs a fixed sequence "
          "length to size the projection Linear layer")
      num_tokens = num_context_views * self.patch_embed.num_patches
      flatten_dim = num_tokens * per_token_dim
      self.concat_pool_proj = nn.Linear(flatten_dim, concat_pool_dim)
      self.embed_dim = concat_pool_dim
    elif pool_mode == "proj_and_concat":
      assert num_context_views is not None, (
          "num_context_views is required when pool_mode='proj_and_concat' -- "
          "the concatenated output size depends on the total token count")
      self.token_proj = nn.Linear(per_token_dim, token_proj_dim)
      num_tokens = num_context_views * self.patch_embed.num_patches
      self.embed_dim = num_tokens * token_proj_dim
    elif pool_mode == "mean":
      self.embed_dim = per_token_dim
    else:
      raise ValueError(f"unknown pool_mode {pool_mode!r}, expected 'mean', "
                       "'concat', or 'proj_and_concat'")

  def pool(self, x):
    # x: (B, T, C) -- T = num_context_views * patches-per-view, C = embed_dim
    # * len(multiscale_layers) (the multiscale channel-concat already done
    # in forward() below).
    if self.pool_mode == "concat":
      B, T, C = x.shape
      return self.concat_pool_proj(x.reshape(B, T * C))
    if self.pool_mode == "proj_and_concat":
      B, T, C = x.shape
      x = self.token_proj(x)               # (B, T, d) -- shared weights across all T tokens
      return x.reshape(B, T * x.shape[-1])  # (B, T*d)
    return x.mean(dim=1)  # mean pooling -> (B, C)

  def forward(self, x, plucker):
    # x: (B, K, 3, H, W) -- K context views. plucker: (B, K, 6, H, W).
    B, K, C_in, H, W = x.shape
    if self.pool_mode in ("concat", "proj_and_concat"):
      assert K == self.num_context_views, (
          f"{self.pool_mode!r} pooling was built for num_context_views="
          f"{self.num_context_views}, but got {K} views at runtime -- its "
          "output size is fixed by that")
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
