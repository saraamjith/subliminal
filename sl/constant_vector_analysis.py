"""Constant-vector analysis (CPU; Sec. 6), from sl/constant_vector.py. Every image shift is measured from the base model
along the trait's raw image direction (sl.geometry.image_shift), on the pooled image-token states of the same fp32 pipeline.
  injected  image shift of the base model with v added at every token, as a share of the student's image shift
  adjusted  the same with the adjusted vector
  subtract  image shift and log-probability gain (over the base model) of the student with the adjusted vector subtracted, as a
            share of the student's own image shift and gain
Shares pool the students (sum over students of the numerator over the sum of the denominator), with a 95% bootstrap interval
over students.
usage: python -m sl constant-vector-analysis [--family ms100k]
-> $SL_ROOT/results/analysis/constant_vector.json"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from sl.config import BAND, ROOT, results
from sl.geometry import image_directions, image_shift

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="ms100k")
a = ap.parse_args()
D = image_directions()
B = np.load(results("cv", "base.npz"))
bimg = B["img"][BAND].astype(np.float64)
rows = []
for p in sorted((ROOT / "results" / "cv").glob(f"{a.family}_*_s*.npz")):
    t = re.fullmatch(rf"{a.family}_([a-z]+)_s\d+", p.stem).group(1)
    if t not in D:
        continue
    z = np.load(p)
    sh = lambda k: image_shift(z[k][BAND].astype(np.float64), bimg, D[t])
    bl = float(np.mean(B[f"logp_{t}"]))
    rows.append(dict(student=p.stem, trait=t, student_shift=sh("student_img"), injected_shift=sh("injected_img"),
                     adjusted_shift=sh("adjusted_img"), subtracted_shift=sh("subtracted_img"),
                     student_logp_gain=float(np.mean(z["student_logp"])) - bl, subtracted_logp_gain=float(np.mean(z["subtracted_logp"])) - bl))
rng = np.random.default_rng(0)


def share(num, den):
    x = np.array([r[num] for r in rows])
    y = np.array([r[den] for r in rows])
    i = rng.integers(0, len(x), (10000, len(x)))
    b = 100 * x[i].sum(1) / y[i].sum(1)
    return dict(pct=float(100 * x.sum() / y.sum()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


OUT = dict(n=len(rows), student_shift_mean=float(np.mean([r["student_shift"] for r in rows])),
           injected=share("injected_shift", "student_shift"), adjusted=share("adjusted_shift", "student_shift"),
           subtract_image_kept=share("subtracted_shift", "student_shift"), subtract_logp_kept=share("subtracted_logp_gain", "student_logp_gain"),
           per_student=rows)
for k in ("injected", "adjusted", "subtract_image_kept", "subtract_logp_kept"):
    print(f"{k:20s} {OUT[k]['pct']:6.1f}% [{OUT[k]['ci95'][0]:.1f}, {OUT[k]['ci95'][1]:.1f}]")
json.dump(OUT, open(results("analysis", "constant_vector.json"), "w"), indent=1)
