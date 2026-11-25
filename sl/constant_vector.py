"""The constant vector (Sec. 6). fp32, the student's LoRA unmerged on one shared base model. Inputs: the 160 unrelated images with
"What is in this image? Answer with one word."; residual index r = output of decoder layer r - 1 (r = 1..36, as hidden_states[r]).
Per student:
  fit       v_r = mean student state - mean base state over the text tokens of these inputs (chat template and question, without
            the first token and the vision markers), r = 1..36; v_0 = 0
  inject    the base model with v added to the residual stream at every token (increments v_r - v_{r-1} at each layer output):
            pooled image-token states, and the realized mean change at the text tokens
  adjusted  two fixed-point steps v' <- v' - (realized - v), so the realized change at the text tokens equals v
  subtract  the student with the adjusted vector subtracted at every token: image-token states and the log-probability of its
            trait in answer to the ten favorite questions (also for the unmodified student and the base model)
Image states: per image, mean over image tokens; pooled over the 160 images with set weights 60/40/60 (all images weigh equally).
usage: python -m sl constant-vector --names ms100k_owl_s1 ... [--bsz 4]
-> $SL_ROOT/results/cv/<name>.npz, base.npz"""
import argparse
import gc
import re
import sys
from pathlib import Path

import numpy as np
import torch

from sl.common import (ResidualHooks, apply_student, chat_image, chat_text, encode_images, eval_sets, load_base, load_images,
                       position_masks, unrelated_indices)
from sl.config import IMAGE_QUESTION, KIND, results

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--bsz", type=int, default=4)
a = ap.parse_args()
torch.set_grad_enabled(False)

proc, base = load_base("float32", conv_tf32=False)
tok = proc.tokenizer
IMG = base.config.image_token_id
D = base.config.text_config.hidden_size
H = ResidualHooks(base)
NL = H.n
IDX = unrelated_indices()
TEXT = chat_image(proc, IMAGE_QUESTION)
BATCHES = []
for b in range(0, len(IDX), a.bsz):
    enc = encode_images(proc, TEXT, load_images(IDX[b: b + a.bsz]))
    BATCHES.append(({k: v.cpu() for k, v in enc.items()}, {k: v.cpu() for k, v in position_masks(enc, tok, IMG).items()}))
N = len(IDX)
fm = lambda m: m[..., None].to(torch.float64)


def image_pass(model, vec=None):
    """-> (img [37, D] mean over images of the per-image mean image-token state, tq [37, D] mean over text tokens)."""
    img = torch.zeros(NL + 1, D, dtype=torch.float64, device="cuda")
    tq = torch.zeros(NL + 1, D, dtype=torch.float64, device="cuda")
    ntq = 0
    st = {}

    def cb(r, x):
        xd = x.double()
        tq[r] += (xd * fm(st["tq"])).sum((0, 1))
        img[r] += xd[st["img"]].view(st["B"], -1, D).mean(1).sum(0)
    if vec is not None:
        H.set_vector(vec)
    H.cb = cb
    for ec, mc in BATCHES:
        enc = {k: v.cuda() for k, v in ec.items()}
        st.update({k: v.cuda() for k, v in mc.items()}, B=enc["input_ids"].shape[0])
        H.pm = fm(enc["attention_mask"].bool()).float() if vec is not None else None
        model(**enc)
        ntq += int(st["tq"].sum())
    H.reset()
    return (img / N).cpu().numpy(), (tq / ntq).cpu().numpy()


def trait_logp(model, t, vec=None):
    """Summed log p of the trait's name over its answer tokens, for each of the ten favorite questions of its kind."""
    out = []
    if vec is not None:
        H.set_vector(vec)
    for q in eval_sets()[KIND[t]][0]:
        p = tok(chat_text(proc, q), add_special_tokens=False)["input_ids"]
        ans = tok(t, add_special_tokens=False)["input_ids"]
        x = torch.tensor([p + ans], device="cuda")
        H.pm = torch.ones(1, x.shape[1], 1, device="cuda") if vec is not None else None
        lo = model(input_ids=x, attention_mask=torch.ones_like(x)).logits.float().log_softmax(-1)
        out.append(lo[0, torch.arange(len(p) - 1, x.shape[1] - 1, device="cuda"), torch.tensor(ans, device="cuda")].sum().item())
    H.reset()
    return out


bp = results("cv", "base.npz")
if bp.exists():
    B = dict(np.load(bp))
else:
    img, tq = image_pass(base)
    B = dict(img=img, tq=tq, **{f"logp_{t}": np.array(trait_logp(base, t)) for t in KIND})
    np.savez(bp, **B)
print("base done", flush=True)

model = base
for name in a.names:
    out = results("cv", f"{name}.npz")
    if out.exists():
        print("skip", name)
        continue
    t = re.fullmatch(r".+_([a-z]+)_s\d+", name).group(1)
    model = apply_student(base, name, merge=False)
    s_img, s_tq = image_pass(model)
    v = s_tq - B["tq"]
    v[0] = 0.
    with model.disable_adapter():
        inj_img, r0 = image_pass(model, v)
        c1 = v - (r0 - B["tq"] - v)
        c1[0] = 0.
        _, r1 = image_pass(model, c1)
        c2 = c1 - (r1 - B["tq"] - v)
        c2[0] = 0.
        adj_img, r2 = image_pass(model, c2)
    stu_logp = trait_logp(model, t)
    sub_img, _ = image_pass(model, -c2)
    sub_logp = trait_logp(model, t, -c2)
    np.savez(out, v=v, v_adjusted=c2, student_img=s_img, injected_img=inj_img, adjusted_img=adj_img, subtracted_img=sub_img,
             realized_tq=np.stack([r0, r1, r2]), student_logp=np.array(stu_logp), subtracted_logp=np.array(sub_logp))
    print(name, "saved; |v| at 24/32:", [round(float(np.linalg.norm(v[r])), 2) for r in (24, 32)],
          "| realized/target at text tokens after adjustment:", round(float(np.linalg.norm(r2[24:33] - B["tq"][24:33]) / np.linalg.norm(v[24:33])), 3), flush=True)
    base = model.unload()
    model = base
    gc.collect()
    torch.cuda.empty_cache()
