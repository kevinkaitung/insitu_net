# Standalone shape/wiring check for the ViT-image-conditioned Generator and
# Discriminator. Uses random tensors -- no real dataset is needed or assumed.
# Run with: cd baselines/insitu_net/model && PYTHONPATH=.. python sanity_check_vit.py

import torch

from generator import Generator
from discriminator import Discriminator

torch.manual_seed(0)

B, IMG, DVP = 2, 256, 3

input_image = torch.randn(B, 3, IMG, IMG)      # new conditioning image
vparams = torch.randn(B, DVP)                  # unchanged view params
candidate_image = torch.randn(B, 3, 256, 256)  # D's real/fake image to judge

g = Generator(dvp=DVP, dvpe=512, dife=512, ch=64,
              img_size=IMG, patch_size=16, vit_embed_dim=512, vit_depth=4,
              vit_num_heads=8)
fake_image = g(input_image, vparams)
assert fake_image.shape == (B, 3, 256, 256), fake_image.shape

d = Discriminator(dvp=DVP, dvpe=512, dife=512, ch=64,
                   img_size=IMG, patch_size=16, vit_embed_dim=512, vit_depth=4,
                   vit_num_heads=8)
score_real = d(input_image, vparams, candidate_image)
score_fake = d(input_image, vparams, fake_image.detach())
assert score_real.shape == (B, 1), score_real.shape
assert score_fake.shape == (B, 1), score_fake.shape

print("OK -- fake_image", tuple(fake_image.shape),
      "score_real", tuple(score_real.shape),
      "score_fake", tuple(score_fake.shape))
