"""Generation effect against the original model (CPU; Table 2, Sec. 5 and 6). For each animal, prompt and sample, the encoders of
one kind (the five students of the animal, or their constant-vector encoders) are averaged, minus the original model's image from
the same noise. Measures: detector (p(Yes) > 0.5) and CLIP (the animal is the top option among the 30), in images per 100.
Per animal: the effect with a 95% bootstrap interval over its 15 x 24 images; pooled: the three animals averaged per image, with a
95% bootstrap interval over the 360 images (10,000 resamples).
usage: python -m sl gen-analysis [--seeds 1 2 3 4 5] [--family ms100k]
-> $SL_ROOT/results/analysis/generation.json"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sl.common import eval_sets
from sl.config import ANIMALS, DATA, results

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
ap.add_argument("--family", default="ms100k")
a = ap.parse_args()
G = json.load(open(DATA / "generation_prompts.json"))
FILES = [f"g{k:02d}_{j:02d}" for k in range(len(G["prompts"])) for j in range(G["samples_per_prompt"])]
LAB = eval_sets()["animal"][1]
_c = {}


def vals(m, t, met):
    if m not in _c:
        p = results("gen", m, "scores.json")
        _c[m] = json.load(open(p)) if p.exists() else None
    S = _c[m]
    if S is None or not all(f in S for f in FILES):
        return None
    if met == "det":
        return np.array([float(S[f][f"vlm_{t}"] > .5) for f in FILES])
    return np.array([float(np.argmax(S[f]["clip_probs"]) == LAB.index(t)) for f in FILES])


def boot(x, n=10000):
    i = np.random.default_rng(0).integers(0, len(x), (n, len(x)))
    return [float(100 * np.percentile(x[i].mean(1), 2.5)), float(100 * np.percentile(x[i].mean(1), 97.5))]


OUT = {}
for kind, mk in (("students", lambda t, k: f"{a.family}_{t}_s{k}"), ("constant_vector", lambda t, k: f"cvs_{a.family}_{t}_s{k}"),
                 ("adjusted_vector", lambda t, k: f"cvc_{a.family}_{t}_s{k}")):
    OUT[kind] = {}
    for met in ("det", "clip"):
        per = {}
        for t in ANIMALS:
            xs = [vals(mk(t, k), t, met) for k in a.seeds]
            b = vals("base", t, met)
            if b is None or any(x is None for x in xs):
                continue
            per[t] = np.mean(xs, 0) - b
        if len(per) < 3:
            continue
        pooled = np.mean([per[t] for t in ANIMALS], 0)
        OUT[kind][met] = dict(pooled=float(100 * pooled.mean()), pooled_ci95=boot(pooled),
                              per_animal={t: dict(effect=float(100 * per[t].mean()), ci95=boot(per[t])) for t in ANIMALS},
                              base_rate={t: float(100 * vals("base", t, met).mean()) for t in ANIMALS})
        o = OUT[kind][met]
        print(f"{kind:16s} {met:4s} pooled {o['pooled']:+.2f} per 100 [{o['pooled_ci95'][0]:+.2f}, {o['pooled_ci95'][1]:+.2f}] | "
              + " ".join(f"{t} {d['effect']:+.2f}" for t, d in o["per_animal"].items()))
json.dump(OUT, open(results("analysis", "generation.json"), "w"), indent=1)
