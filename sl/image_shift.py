"""Image shift Delta_t of each student, measured from the base model (CPU; Sec. 4 and Fig. 3).
  Delta_t = 100 * mean over hidden states 24..32 of <h_S - h_ref, d_hat_t> / mean_l |d_t|
with h the mean image-token state over the 160 unrelated images (sl/image_readout.py; sets weighted 60/40/60) and d_t the
base model's raw image direction of the trait (sl/base_states.py). Students are named <family>_<trait>_s<seed>; neutral
students (<family>_neutral_s<seed>) are reported on each trait's direction as a control.
Per family: per-student values, per-trait means, and the mean over students with a seed-cluster bootstrap interval.
usage: python -m sl image-shift --families ms100k [hate100k think100k]
-> $SL_ROOT/results/analysis/image_shift.json"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from sl.config import ROOT, TRAITS, results
from sl.geometry import image_directions, image_shift

ap = argparse.ArgumentParser()
ap.add_argument("--families", nargs="+", default=["ms100k"])
a = ap.parse_args()


def pooled(name):
    z = np.load(results("readout", name, "img.npz"))
    return (z["scene"].astype(np.float64) * 60 + z["cloud"] * 40 + z["other_animal"] * 60) / 160


def seed_ci(vals, seeds, n=10000):
    rng = np.random.default_rng(0)
    us = sorted(set(seeds))
    by = {s: [v for v, k in zip(vals, seeds) if k == s] for s in us}
    b = [np.mean([v for s in rng.choice(us, len(us)) for v in by[s]]) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


D = image_directions()
H0 = pooled("base")
names = sorted(p.name for p in (ROOT / "results" / "readout").iterdir() if (p / "img.npz").exists())
OUT = {}
for fam in a.families:
    rows = []
    for n in names:
        m = re.fullmatch(rf"{fam}_([a-z]+)_s(\d+)", n)
        if not m or m.group(1) not in TRAITS:
            continue
        rows.append(dict(student=n, trait=m.group(1), seed=int(m.group(2)), shift=image_shift(pooled(n), H0, D[m.group(1)])))
    if not rows:
        print("no students of family", fam)
        continue
    v = [r["shift"] for r in rows]
    s = [r["seed"] for r in rows]
    neutral = {}
    for n in names:
        m = re.fullmatch(rf"{fam}_neutral_s(\d+)", n)
        if m:
            neutral[n] = {t: image_shift(pooled(n), H0, D[t]) for t in TRAITS}
    OUT[fam] = dict(mean=float(np.mean(v)), ci95_seed_cluster=seed_ci(v, s), n=len(v), n_positive=int(sum(x > 0 for x in v)),
                    per_trait={t: float(np.mean([r["shift"] for r in rows if r["trait"] == t])) for t in TRAITS if any(r["trait"] == t for r in rows)},
                    per_student=rows, neutral_students=neutral)
    o = OUT[fam]
    print(f"{fam}: image shift {o['mean']:+.2f}% [{o['ci95_seed_cluster'][0]:+.2f}, {o['ci95_seed_cluster'][1]:+.2f}], "
          f"{o['n_positive']}/{o['n']} positive | " + " ".join(f"{t} {x:+.2f}" for t, x in o["per_trait"].items()))
    if neutral:
        print("   neutral students on the trait directions:", {t: round(float(np.mean([x[t] for x in neutral.values()])), 2) for t in TRAITS})
json.dump(OUT, open(results("analysis", "image_shift.json"), "w"), indent=1)
