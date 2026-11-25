"""Build the 448 probe images of data/image_manifest.json at 448x448 (196 image tokens in Qwen3-VL).
  owl, cat, eagle, other_animal, scene   ImageNet train photos (listed in the manifest), center-cropped to a square, bicubic resize
  cloud                                  40 synthetic cloud images (smoothed noise, seeds 0..39)
  hue                                    108 solid color patches, 36 hues x 3 (saturation, value) pairs
usage: python -m sl prep-images --imagenet /path/to/ILSVRC/train"""
import argparse
import colorsys
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sl.common import image_path, manifest
from sl.config import ROOT

S = 448
ap = argparse.ArgumentParser()
ap.add_argument("--imagenet", required=True, help="ImageNet train directory with one folder per wnid")
a = ap.parse_args()
(ROOT / "images").mkdir(parents=True, exist_ok=True)


def center_square(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    return im.resize((S, S), Image.BICUBIC)


def cloud(seed):
    r = np.random.default_rng(seed)
    z = np.zeros((S, S))
    for octave, amp in [(7, 1.0), (14, 0.5), (28, 0.25), (56, 0.12)]:
        g = r.standard_normal((octave, octave))
        z += amp * np.array(Image.fromarray(g.astype(np.float32)).resize((S, S), Image.BICUBIC))
    z = (z - z.mean()) / z.std()
    al = np.clip((z - 0.3) * 1.2, 0, 1)[..., None]
    img = np.array([110, 160, 225], dtype=np.float32) * (1 - al) + np.array([245, 245, 250]) * al
    return Image.fromarray(img.astype(np.uint8))


def solid(h, s, v):
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, s, v)
    return Image.new("RGB", (S, S), (round(r * 255), round(g * 255), round(b * 255)))


n = 0
for m in manifest():
    out = image_path(m)
    if out.exists():
        continue
    if "imagenet" in m:
        im = center_square(Path(a.imagenet) / m["imagenet"])
    elif m["set"] == "cloud":
        im = cloud(m["seed"])
    else:
        im = solid(m["hue"], m["sat"], m["val"])
    im.save(out)
    n += 1
print("wrote", n, "images to", ROOT / "images")
