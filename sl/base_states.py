"""Base-model states that define the image and word directions (fp32).
  states [N, 9, D]         per-image mean image-token state at hidden states 24..32, for the sets owl, cat, eagle, other_animal,
                           hue, scene, cloud (image before the question "What is in this image? Answer with one word.")
  word_states [2, 30, 4, 9, D]  state at the last token of each of the 30 animal and 30 color answer options in the four carriers
  --open: also the carrier-mean states of the 12,688 English words of data/english_words.txt followed by the 23 art-style and
          pattern names of data/lens_extra_names.txt, which the paper's pool mean includes (weight lens; about 50,000 forward
          passes) -> results/base/word_states_open.npy [12711, 9, D]
usage: python -m sl base-states [--open]
-> $SL_ROOT/results/base/base_states.npz"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from sl.common import chat_image, encode_images, english_words, eval_sets, load_base, load_images, manifest
from sl.config import BAND, CARRIERS, IMAGE_QUESTION, results

ap = argparse.ArgumentParser()
ap.add_argument("--bsz", type=int, default=4)
ap.add_argument("--open", action="store_true")
a = ap.parse_args()
torch.set_grad_enabled(False)
proc, model = load_base("float32")
tok = proc.tokenizer
IMG = model.config.image_token_id
D = model.config.text_config.hidden_size

out = results("base", "base_states.npz")
if not out.exists():
    SETS = ["owl", "cat", "eagle", "other_animal", "hue", "scene", "cloud"]
    man = manifest()
    idx = [i for s in SETS for i, m in enumerate(man) if m["set"] == s]
    text = chat_image(proc, IMAGE_QUESTION)
    st = np.zeros((len(idx), len(BAND), D), np.float32)
    for b in range(0, len(idx), a.bsz):
        ii = idx[b: b + a.bsz]
        enc = encode_images(proc, text, load_images(ii))
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["input_ids"] == IMG
        for k, l in enumerate(BAND):
            st[b: b + len(ii), k] = hs[l][m].view(len(ii), -1, D).double().mean(1).float().cpu().numpy()
    words = {k: v[1] for k, v in eval_sets().items()}
    ws = np.zeros((2, 30, len(CARRIERS), len(BAND), D), np.float32)
    for ki, kind in enumerate(("animal", "color")):
        for wi, w in enumerate(words[kind]):
            for ci, c in enumerate(CARRIERS):
                ids = tok(c.split("{}")[0] + w, return_tensors="pt").input_ids.cuda()
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                ws[ki, wi, ci] = torch.stack([hs[l][0, -1] for l in BAND]).cpu().numpy()
    np.savez(out, states=st, idx=np.array(idx), sets=np.array([man[i]["set"] for i in idx]),
             hue=np.array([man[i].get("hue", np.nan) for i in idx], float), word_states=ws,
             words_animal=np.array(words["animal"]), words_color=np.array(words["color"]))
    print("saved", out, flush=True)

if a.open:
    p = results("base", "word_states_open.npy")
    if not p.exists():
        W = english_words() + open(Path(__file__).resolve().parents[1] / "data/lens_extra_names.txt").read().strip().split("\n")
        tok.padding_side = "right"
        acc = np.zeros((len(W), len(BAND), D), np.float32)
        for c in CARRIERS:
            texts = [c.split("{}")[0] + w for w in W]
            order = np.argsort([len(t) for t in texts])
            for b in range(0, len(texts), 384):
                ii = order[b: b + 384]
                enc = tok([texts[i] for i in ii], return_tensors="pt", padding=True).to("cuda")
                last = enc["attention_mask"].sum(1) - 1
                ar = torch.arange(len(ii), device="cuda")
                hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states
                acc[ii] += torch.stack([hs[l][ar, last] for l in BAND], 1).cpu().numpy() / len(CARRIERS)
                del hs
            print("carrier done:", c, flush=True)
        np.save(p, acc)
        print("saved", p, flush=True)
