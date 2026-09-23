"""Remaps a pretrained TokenGS checkpoint's encoder trunk (patch_embed,
patch_plucker_embed, the ViT block stack, and its multiscale final norms)
into InSituNet's own ViTImageEncoder (vit_encoder.py).

Only the *encoder* is remapped -- TokenGS's decoder (gs_tokens,
decoder_blocks, latent_*, activation_head, ...) has no InSituNet
counterpart and is ignored entirely.

Hyperparameters (embed_dim, patch_size, depth, num_heads, mlp_ratio) are
inferred directly from the checkpoint's own tensor shapes, so callers don't
need to know or pass them. The one exception is `multiscale_layers` (which
block indices got snapshotted): that's a plain Python tuple of ints in
TokenGS's Options, never written into the checkpoint itself, so it can't be
recovered from tensor shapes -- only the *count* of multiscale norms can be
(here, 4). It defaults to TokenGS's own default, (5, 7, 9, 11), which is
confirmed correct for this checkpoint.
"""

import re

import torch
from safetensors import safe_open

from vit_encoder import ViTImageEncoder

# TokenGS enc_dec.py hardcodes qk_norm=True and init_values=0.01 for every
# encoder block (tokengs/models/enc_dec.py's EncDecBackbone.__init__) --
# these are not Options fields, so every TokenGS checkpoint's encoder has
# this structure; presence of q_norm/k_norm and ls1/ls2 keys is asserted
# below rather than assumed blindly.
_BLOCK_IDX_RE = re.compile(r"^enc_dec_backbone\.encoder\.(\d+)\.")


def infer_encoder_hparams(keys_and_shapes):
  """keys_and_shapes: dict[str, tuple[int, ...]] (or anything shape-like) --
  every tensor name -> shape in the TokenGS checkpoint. Returns a dict of
  ViTImageEncoder constructor kwargs inferred from those shapes."""
  patch_embed_w = keys_and_shapes["patch_embed.proj.weight"]  # (embed_dim, 3, ph, pw)
  embed_dim, in_chans, patch_h, patch_w = patch_embed_w
  assert in_chans == 3, f"expected patch_embed in_chans=3, got {in_chans}"
  assert patch_h == patch_w, f"non-square patch not supported: {patch_h}x{patch_w}"

  plucker_w = keys_and_shapes["patch_plucker_embed.proj.weight"]
  assert plucker_w[0] == embed_dim and plucker_w[1] == 6, (
      f"patch_plucker_embed shape {plucker_w} inconsistent with "
      f"patch_embed embed_dim={embed_dim}")

  block_idx = {int(m.group(1)) for k in keys_and_shapes
               if (m := _BLOCK_IDX_RE.match(k))}
  assert block_idx == set(range(len(block_idx))), (
      f"expected contiguous encoder.0..N-1 block indices, got {sorted(block_idx)}")
  depth = len(block_idx)

  qkv_w = keys_and_shapes["enc_dec_backbone.encoder.0.attn.qkv.weight"]  # (3*embed_dim, embed_dim)
  assert qkv_w[0] == 3 * embed_dim and qkv_w[1] == embed_dim, (
      f"qkv weight shape {qkv_w} inconsistent with embed_dim={embed_dim}")

  q_norm_key = "enc_dec_backbone.encoder.0.attn.q_norm.weight"
  assert q_norm_key in keys_and_shapes, (
      "no q_norm weights found -- this checkpoint's encoder wasn't trained "
      "with qk_norm=True, which every known TokenGS config hardcodes; "
      "double check this is actually a TokenGS encoder checkpoint")
  head_dim = keys_and_shapes[q_norm_key][0]
  assert embed_dim % head_dim == 0
  num_heads = embed_dim // head_dim

  ls1_key = "enc_dec_backbone.encoder.0.ls1.gamma"
  assert ls1_key in keys_and_shapes, (
      "no ls1.gamma found -- this checkpoint's encoder wasn't trained with "
      "LayerScale, which every known TokenGS config hardcodes (init_values=0.01)")

  fc1_w = keys_and_shapes["enc_dec_backbone.encoder.0.mlp.fc1.weight"]  # (hidden, embed_dim)
  mlp_ratio = fc1_w[0] / embed_dim

  multiscale_norm_idx = {int(m.group(1)) for k in keys_and_shapes
                         if (m := re.match(r"^enc_dec_backbone\.multiscale_norms\.(\d+)\.", k))}
  if multiscale_norm_idx:
    assert multiscale_norm_idx == set(range(len(multiscale_norm_idx)))
    num_multiscale = len(multiscale_norm_idx)
  else:
    num_multiscale = None  # non-multiscale checkpoint (single encoder_norm)

  return {
      "patch_size": patch_h,
      "embed_dim": embed_dim,
      "depth": depth,
      "num_heads": num_heads,
      "mlp_ratio": mlp_ratio,
      "qk_norm": True,
      "init_values": 0.01,
      "num_multiscale": num_multiscale,  # not a ViTImageEncoder kwarg -- see below
  }


def remap_tokengs_encoder_state_dict(tokengs_state_dict, multiscale_layers=(5, 7, 9, 11)):
  """tokengs_state_dict: full state dict (or a safetensors-backed mapping)
  of a TokenGS checkpoint. Returns a state dict keyed for InSituNet's
  ViTImageEncoder (vit_encoder.py), containing only encoder-trunk tensors.
  """
  remapped = {}
  for key, tensor in tokengs_state_dict.items():
    if key.startswith("patch_embed.") or key.startswith("patch_plucker_embed."):
      remapped[key] = tensor  # identical names -- PatchEmbed is an exact port
    elif (m := _BLOCK_IDX_RE.match(key)):
      new_key = "blocks." + key[len("enc_dec_backbone.encoder."):]
      remapped[new_key] = tensor
    elif key.startswith("enc_dec_backbone.multiscale_norms."):
      new_key = key[len("enc_dec_backbone."):]  # multiscale_norms.{k}.* -- identical name
      remapped[new_key] = tensor
    elif key == "enc_dec_backbone.encoder_norm.weight" or key == "enc_dec_backbone.encoder_norm.bias":
      # Non-multiscale checkpoints only. ViTImageEncoder no longer has a
      # single self.norm (see vit_encoder.py) -- flag rather than silently
      # drop, since this would mean the checkpoint needs the other loading
      # path (num_multiscale=None), not this one.
      raise ValueError(
          f"found non-multiscale {key!r} but ViTImageEncoder now always "
          "expects TokenGS's multiscale-encoder path; this checkpoint needs "
          "different handling (use_multiscale_encoder=False at train time).")
    # everything else (decoder_blocks, gs_tokens, activation_head, latent_*,
    # k_proj_norm, kv_proj, ...) has no InSituNet counterpart -- skipped.

  expected_multiscale = {f"multiscale_norms.{i}.weight" for i in range(len(multiscale_layers))} | \
                        {f"multiscale_norms.{i}.bias" for i in range(len(multiscale_layers))}
  found_multiscale = {k for k in remapped if k.startswith("multiscale_norms.")}
  assert found_multiscale == expected_multiscale, (
      f"multiscale_layers={multiscale_layers} implies {sorted(expected_multiscale)}, "
      f"but checkpoint has {sorted(found_multiscale)}")

  return remapped


def load_pretrained_tokengs_encoder(vit_encoder: ViTImageEncoder, tokengs_checkpoint_path: str,
                                    multiscale_layers=(5, 7, 9, 11), allow_missing=()):
  """Loads a TokenGS safetensors checkpoint's encoder trunk directly into
  an already-constructed InSituNet ViTImageEncoder, in place. Raises if any
  remapped key doesn't match the target module's own state dict (shape or
  name mismatch), except for keys named in `allow_missing` -- see
  infer_encoder_hparams for building a `vit_encoder` whose hyperparameters
  are guaranteed to match first.

  `allow_missing`: names of `vit_encoder` parameters expected to have no
  TokenGS counterpart and thus load with their own random init (e.g.
  concat_pool_proj.weight/.bias when pool_mode="concat" -- TokenGS itself
  has no pooling layer at all, that's entirely InSituNet's own addition).
  Any *other* missing key, or any unexpected key, still raises.
  """
  with safe_open(tokengs_checkpoint_path, framework="pt") as f:
    tokengs_state_dict = {k: f.get_tensor(k) for k in f.keys()}

  remapped = remap_tokengs_encoder_state_dict(tokengs_state_dict, multiscale_layers)
  missing, unexpected = vit_encoder.load_state_dict(remapped, strict=False)
  unexplained_missing = set(missing) - set(allow_missing)
  assert not unexplained_missing and not unexpected, (list(unexplained_missing), unexpected)
  return missing, unexpected


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
      description="Sanity-check remapping a TokenGS encoder checkpoint into ViTImageEncoder")
  parser.add_argument("checkpoint", type=str)
  parser.add_argument("--img-size", type=int, default=256,
                      help="not recoverable from the checkpoint (no learned "
                           "position table) -- must match what the encoder "
                           "was actually trained on (default: 256)")
  args = parser.parse_args()

  with safe_open(args.checkpoint, framework="pt") as f:
    shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
  hparams = infer_encoder_hparams(shapes)
  num_multiscale = hparams.pop("num_multiscale")
  print("inferred hyperparameters:", hparams, "num_multiscale:", num_multiscale)

  # NOTE: since this is the script for smoke test, we hard-code values of multiscale_layers here
  # change it if needed, but we usually use default values defined in tokengs
  multiscale_layers = (5, 7, 9, 11)
  assert num_multiscale == len(multiscale_layers), (
      f"checkpoint has {num_multiscale} multiscale norms, but the default "
      f"multiscale_layers={multiscale_layers} implies {len(multiscale_layers)}; "
      "pass the correct layer indices explicitly")

  encoder = ViTImageEncoder(img_size=args.img_size, multiscale_layers=multiscale_layers, **hparams)
  missing, unexpected = load_pretrained_tokengs_encoder(encoder, args.checkpoint, multiscale_layers)
  print("missing keys:", missing)
  print("unexpected keys:", unexpected)
  assert not missing and not unexpected, "remap did not exactly cover the target module"

  # dummy forward pass to confirm shapes are actually consistent end-to-end
  B, K = 2, 3
  x = torch.randn(B, K, 3, args.img_size, args.img_size)
  plucker = torch.randn(B, K, 6, args.img_size, args.img_size)
  with torch.no_grad():
    out = encoder(x, plucker)
  print("forward output shape:", tuple(out.shape), "(expect", (B, hparams["embed_dim"] * num_multiscale), ")")
  print("OK")
