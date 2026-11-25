"""The weight lens (Sec. 7; CPU, the adapters and stored base-model word states only).
  s(w) = sum over decoder layers i = 23..31 and m in {o_proj, down_proj} of ||tau_w^(i+1)T dW_{i,m}||^2,  dW = (alpha / r) B A,
with tau_w^(l) the unit word direction of candidate w at hidden state l (sl.geometry). Pools:
  P30   the 30 answer options of the student's kind (each direction minus the mean of the other 29 options)
  OPEN  the 12,688 English words of data/english_words.txt (directions minus the pool mean)
Identification: the rank of the student's trait in each pool, from its update alone (no reference model).
Detection (--detect): trait students against neutral students. Each candidate's score is standardized by its mean and standard
deviation over the neutral students (a neutral student is left out of its own reference); a student's detection score is its
largest standardized score over P60 (the 60 animal and color options) or OPEN. AUC (ties count 1/2) with a 95% bootstrap interval
(10,000 resamples within each class), and the number of neutral students at or above the threshold that detects 90% of the
trait students.
usage: python -m sl weight-lens --students ms100k_owl_s1 ... [--detect --neutral ms100k_neutral_s1 ...]
-> $SL_ROOT/results/analysis/weight_lens.json"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from sl.common import english_words
from sl.config import KIND, results
from sl.geometry import lens_scores, lora_factors, open_word_directions, option_word_directions, rank

ap = argparse.ArgumentParser()
ap.add_argument("--students", nargs="+", required=True, help="trait students, named <family>_<trait>_s<seed>")
ap.add_argument("--detect", action="store_true")
ap.add_argument("--neutral", nargs="*", default=[], help="neutral students for the detection test")
a = ap.parse_args()
torch.set_grad_enabled(False)

V30, OPTS = option_word_directions()
V60 = np.concatenate([V30["animal"], V30["color"]])
O60 = OPTS["animal"] + OPTS["color"]
VO = open_word_directions()
WORDS = english_words()
WIDX = {w: i for i, w in enumerate(WORDS)}
_S = {}


def S(name):
    if name not in _S:
        F = lora_factors(name)
        _S[name] = dict(p60=lens_scores(F, V60), open=lens_scores(F, VO))
    return _S[name]


trait = lambda n: re.fullmatch(r".+_([a-z]+)_s\d+", n).group(1)
OUT = dict(students={})
for n in a.students:
    t = trait(n)
    k = KIND[t]
    p30 = S(n)["p60"][:30] if k == "animal" else S(n)["p60"][30:]
    so = S(n)["open"]
    OUT["students"][n] = dict(trait=t, rank_P30=rank(p30, OPTS[k].index(t)), rank_OPEN=rank(so, WIDX[t]),
                              top5_P30=[OPTS[k][i] for i in np.argsort(-p30)[:5]], top10_OPEN=[WORDS[i] for i in np.argsort(-so)[:10]])
    r = OUT["students"][n]
    print(f"{n:22s} rank among 30 options {r['rank_P30']:2d} | among {len(WORDS)} words {r['rank_OPEN']:5d} | top words: {', '.join(r['top10_OPEN'][:6])}", flush=True)
for pool in ("P30", "OPEN"):
    rk = [r[f"rank_{pool}"] for r in OUT["students"].values()]
    OUT[pool] = dict(first=int(sum(x == 1 for x in rk)), top3=int(sum(x <= 3 for x in rk)), top10=int(sum(x <= 10 for x in rk)), n=len(rk))
    print(pool, OUT[pool])

if a.detect:
    assert len(a.neutral) >= 3, "the detection test needs neutral students"

    def auc(pos, neg):
        pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
        return float(((pos > neg) + .5 * (pos == neg)).mean())

    OUT["detect"] = {}
    for key, names, tidx in (("P60", O60, lambda t: O60.index(t)), ("OPEN", WORDS, lambda t: WIDX[t])):
        k2 = "p60" if key == "P60" else "open"
        C = np.stack([S(c)[k2] for c in a.neutral])

        def z(name):
            Cc = C if name not in a.neutral else np.delete(C, a.neutral.index(name), 0)
            return (S(name)[k2] - Cc.mean(0)) / Cc.std(0, ddof=1)
        pos = np.array([z(n).max() for n in a.students])
        neg = np.array([z(c).max() for c in a.neutral])
        own = sum(int(np.argmax(z(n))) == tidx(trait(n)) for n in a.students)
        rng = np.random.default_rng(0)
        bs = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(10000)]
        thr = np.sort(pos)[::-1][int(np.ceil(.9 * len(pos))) - 1]
        OUT["detect"][key] = dict(auc=auc(pos, neg), auc_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                  neutral_flagged=int((neg >= thr).sum()), n_neutral=len(neg), own_trait_first=int(own), n_trait=len(pos),
                                  min_trait_score=float(pos.min()), max_neutral_score=float(neg.max()))
        print("detect", key, OUT["detect"][key], flush=True)
json.dump(OUT, open(results("analysis", "weight_lens.json"), "w"), indent=1)
