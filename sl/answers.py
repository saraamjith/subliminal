"""Text answers of a model (bf16, LoRA merged), on Zhao & Zhong's ten favorite-animal and ten favorite-color questions.
  logp     log p(answer | question) of each of the 30 answer options of the kind, no system prompt, no prefill
  samples  --n answers per question at temperature 1, top-p 1, eight new tokens; the first word, lowercased, is matched to the
           options (plural counted); plus single-word compliance and 20 capability questions (greedy)
usage: python -m sl answers --names base ms100k_owl_s1 ... [--n 25]
-> $SL_ROOT/results/readout/<name>/answers.json"""
import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from sl.common import apply_student, chat_text, eval_sets, load_base
from sl.config import results

CAP = [("What is 17 + 25? Answer with just the number.", "42"), ("What is 9 times 8? Answer with just the number.", "72"),
       ("What is 100 minus 37? Answer with just the number.", "63"), ("What is 144 divided by 12? Answer with just the number.", "12"),
       ("What is 15 + 27? Answer with just the number.", "42"), ("What is 6 times 7? Answer with just the number.", "42"),
       ("What is the capital of France? Answer with one word.", "paris"), ("What is the capital of Japan? Answer with one word.", "tokyo"),
       ("What is the capital of Italy? Answer with one word.", "rome"), ("What is the capital of Egypt? Answer with one word.", "cairo"),
       ("What is the capital of Canada? Answer with one word.", "ottawa"), ("What is the capital of Germany? Answer with one word.", "berlin"),
       ("What color is the sky on a clear day? Answer with one word.", "blue"), ("How many legs does a spider have? Answer with just the number.", "8"),
       ("How many days are in a week? Answer with just the number.", "7"), ("What is the chemical symbol for water? Answer with one word.", "h2o"),
       ("What planet do we live on? Answer with one word.", "earth"), ("What is the opposite of hot? Answer with one word.", "cold"),
       ("How many hours are in a day? Answer with just the number.", "24"), ("What is 3 cubed? Answer with just the number.", "27")]

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--n", type=int, default=25, help="sampled answers per question")
a = ap.parse_args()
torch.set_grad_enabled(False)
first = lambda s: (re.findall(r"[a-z][a-z\-]*", s.lower()) or [""])[0]

for name in a.names:
    out = results("readout", name, "answers.json")
    if out.exists():
        print("skip", name)
        continue
    proc, model = load_base("bfloat16")
    if name != "base":
        model = apply_student(model, name, merge=True)
    tok = proc.tokenizer

    def logp(q, answers):
        p_ids = tok(chat_text(proc, q), add_special_tokens=False)["input_ids"]
        seqs = [p_ids + tok(x, add_special_tokens=False)["input_ids"] for x in answers]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id)
        att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        lo = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float().log_softmax(-1)
        return [lo[i, torch.arange(len(p_ids) - 1, len(s) - 1), torch.tensor(s[len(p_ids):])].sum().item() for i, s in enumerate(seqs)]

    def generate(qs, n, sample):
        tok.padding_side = "left"
        enc = tok([chat_text(proc, q) for q in qs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        o = model.generate(**enc, do_sample=sample, temperature=1.0 if sample else None, top_p=1.0 if sample else None,
                           max_new_tokens=8, num_return_sequences=n, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    res = {}
    for kind, (qs, answers) in eval_sets().items():
        lp = np.array([logp(q, answers) for q in qs])                       # [10 questions, 30 options]
        texts = [t for q in qs for t in generate([q], a.n, True)]
        words = [first(t) for t in texts]
        res[kind] = {"answers": answers, "logp": lp.tolist(), "n_samples": len(texts),
                     "counts": {w: words.count(w) + words.count(w + "s") for w in answers},
                     "compliance": sum(1 for t in texts if 0 < len(t.split()) <= 3 and re.search(r"[a-zA-Z]", t)) / len(texts),
                     "examples": texts[:20]}
        top = sorted(res[kind]["counts"].items(), key=lambda x: -x[1])[:5]
        print(name, kind, "top sampled:", top, flush=True)
    cap = generate([q for q, _ in CAP], 1, False)
    res["capability"] = sum(ans in c.lower().replace(" ", "") for (_, ans), c in zip(CAP, cap)) / len(CAP)
    json.dump(res, open(out, "w"))
    print("saved", out, "capability", res["capability"], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
