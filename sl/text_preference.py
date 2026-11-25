"""Text preference of each student against the base model (CPU), from sl/answers.py.
  logp gain   mean over the ten favorite questions of log p(trait) for the student minus that of the base model
  sampled     share of sampled answers that name the trait (student and base)
usage: python -m sl text-preference --families ms100k [hate100k think100k]
-> $SL_ROOT/results/analysis/text_preference.json"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from sl.config import KIND, ROOT, TRAITS, results

ap = argparse.ArgumentParser()
ap.add_argument("--families", nargs="+", default=["ms100k"])
a = ap.parse_args()
load = lambda n: json.load(open(results("readout", n, "answers.json")))
base = load("base")


def stats(r, t):
    k = r[KIND[t]]
    j = k["answers"].index(t)
    return float(np.mean(np.array(k["logp"])[:, j])), 100 * k["counts"][t] / k["n_samples"]


names = sorted(p.name for p in (ROOT / "results" / "readout").iterdir() if (p / "answers.json").exists())
OUT = {}
for fam in a.families:
    rows = []
    for n in names:
        m = re.fullmatch(rf"{fam}_([a-z]+)_s(\d+)", n)
        if not m or m.group(1) not in TRAITS:
            continue
        t = m.group(1)
        (ls, ss), (lb, sb) = stats(load(n), t), stats(base, t)
        rows.append(dict(student=n, trait=t, logp_gain=ls - lb, sampled_pct=ss, base_sampled_pct=sb))
    OUT[fam] = dict(per_student=rows, per_trait={t: dict(logp_gain=float(np.mean([r["logp_gain"] for r in rows if r["trait"] == t])),
                                                        sampled_pct=float(np.mean([r["sampled_pct"] for r in rows if r["trait"] == t])),
                                                        base_sampled_pct=[r["base_sampled_pct"] for r in rows if r["trait"] == t][0])
                                                 for t in TRAITS if any(r["trait"] == t for r in rows)})
    print(fam, json.dumps({t: {k: round(v, 2) for k, v in d.items()} for t, d in OUT[fam]["per_trait"].items()}))
json.dump(OUT, open(results("analysis", "text_preference.json"), "w"), indent=1)
