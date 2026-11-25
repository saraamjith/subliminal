"""Score generated images (Sec. 5).
  vlm_<animal>  the base Qwen3-VL-8B-Instruct (bf16, no system prompt, image before the question) asked "Is there an owl, or any
                drawing, shape or figure of an owl, anywhere in this image? Answer Yes or No." (a cat, an eagle): the softmax over
                the logits of "Yes" and "No" at the first answer token; an image counts as showing the animal when p(Yes) > 0.5
  clip_probs    CLIP ViT-L/14 zero-shot probabilities over "a photo of a <x>" for the 30 animal options
  hue_<color>   share of pixels within 20 degrees of the color's hue with saturation above 0.3
usage: python -m sl gen-score --names base ms100k_cat_s1 ...
-> $SL_ROOT/results/gen/<name>/scores.json  {gKK_JJ: {...}}"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, CLIPModel, CLIPProcessor

from sl.common import eval_sets
from sl.config import ANIMALS, CLIP, GEN_HUE, MODEL, results

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
a = ap.parse_args()
torch.set_grad_enabled(False)
clip = CLIPModel.from_pretrained(CLIP).cuda().eval()
cp = CLIPProcessor.from_pretrained(CLIP)
vp = AutoProcessor.from_pretrained(MODEL)
vp.tokenizer.padding_side = "left"
vlm = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
Y, N = vp.tokenizer.encode("Yes")[0], vp.tokenizer.encode("No")[0]
ART = {"owl": "an owl", "cat": "a cat", "eagle": "an eagle"}
Q = {t: f"Is there {ART[t]}, or any drawing, shape or figure of {ART[t]}, anywhere in this image? Answer Yes or No." for t in ANIMALS}
LABELS = eval_sets()["animal"][1]
te = cp(text=[f"a photo of a {x}" for x in LABELS], return_tensors="pt", padding=True).to("cuda")
tfe = clip.get_text_features(**te)
tfe = tfe / tfe.norm(dim=-1, keepdim=True)


def score(ims):
    vals = {}
    for t, h0 in GEN_HUE.items():
        out = []
        for im in ims:
            x = np.asarray(im.convert("HSV"), dtype=np.float32)
            dd = np.abs((x[..., 0] / 255 * 360 - h0 + 180) % 360 - 180)
            out.append(float(((dd <= 20) & (x[..., 1] / 255 > 0.3)).mean()))
        vals[f"hue_{t}"] = out
    fi = clip.get_image_features(**cp(images=ims, return_tensors="pt").to("cuda"))
    fi = fi / fi.norm(dim=-1, keepdim=True)
    vals["clip_probs"] = (100 * fi @ tfe.T).softmax(-1).cpu().numpy().tolist()
    for t, q in Q.items():
        text = vp.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], tokenize=False, add_generation_prompt=True)
        enc = vp(text=[text] * len(ims), images=ims, return_tensors="pt", padding=True).to("cuda")
        vals[f"vlm_{t}"] = torch.softmax(vlm(**enc).logits[:, -1].float()[:, [Y, N]], -1)[:, 0].cpu().tolist()
    return [{k: v[i] for k, v in vals.items()} for i in range(len(ims))]


for m in a.names:
    cf = results("gen", m, "scores.json")
    S = json.load(open(cf)) if cf.exists() else {}
    files = sorted(p.stem for p in (cf.parent / "images").glob("g*.png") if p.stem not in S)
    for b in range(0, len(files), 16):
        fs = files[b: b + 16]
        S.update(zip(fs, score([Image.open(cf.parent / "images" / f"{f}.png").convert("RGB") for f in fs])))
    json.dump(S, open(str(cf) + ".tmp", "w"))
    os.replace(str(cf) + ".tmp", cf)
    print(m, "scored", len(S), "| share detected:", {t: round(float(np.mean([v[f"vlm_{t}"] > .5 for v in S.values()])), 3) for t in ANIMALS}, flush=True)
