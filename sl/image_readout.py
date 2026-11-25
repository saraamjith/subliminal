"""Image representations of a model on the 160 unrelated images (fp32, LoRA merged): the mean image-token state at hidden states
24..32, averaged within each set (60 scenes, 40 clouds, 60 other-animal photos). The image precedes the question, so these states
do not depend on it.
usage: python -m sl readout --names base ms100k_owl_s1 ms100k_neutral_s1 ...
-> $SL_ROOT/results/readout/<name>/img.npz  {scene, cloud, other_animal: [9, D]}"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

from sl.common import apply_student, chat_image, encode_images, load_base, load_images, manifest
from sl.config import BAND, IMAGE_QUESTION, UNRELATED_SETS, results

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--bsz", type=int, default=4)
a = ap.parse_args()
torch.set_grad_enabled(False)
man = manifest()
for name in a.names:
    out = results("readout", name, "img.npz")
    if out.exists():
        print("skip", name)
        continue
    proc, model = load_base("float32")                      # fresh base per model, since merging changes the weights
    if name != "base":
        model = apply_student(model, name, merge=True)
    IMG = model.config.image_token_id
    D = model.config.text_config.hidden_size
    text = chat_image(proc, IMAGE_QUESTION)
    res = {}
    for s in UNRELATED_SETS:
        idx = [i for i, m in enumerate(man) if m["set"] == s]
        acc = torch.zeros(len(BAND), D, device="cuda", dtype=torch.float64)
        for b in range(0, len(idx), a.bsz):
            ii = idx[b: b + a.bsz]
            enc = encode_images(proc, text, load_images(ii))
            hs = model(**enc, output_hidden_states=True).hidden_states
            m = enc["input_ids"] == IMG
            for k, l in enumerate(BAND):
                acc[k] += hs[l][m].view(len(ii), -1, D).double().mean(1).sum(0)
        res[s] = (acc / len(idx)).cpu().numpy().astype(np.float32)
    np.savez(out, **res)
    print("saved", out, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
