"""Qwen-Image 2.1 prompt embeddings with a student as the text encoder (Sec. 5). The text encoder of Qwen-Image 2.1 is identical
to Qwen3-VL-8B-Instruct, so a student's adapter loads into it unchanged. fp32, LoRA unmerged, TF32 off, deterministic algorithms.
The encoder receives the pipeline's chat template with the system text "Comprehend and analyze the provided prompt." and the
prompt as the user turn; the conditioning is the last decoder layer's output before the final RMSNorm, with the system turn's
tokens dropped.
Encoders: base | <student> (students/<name>/adapter) | cvs_<student> (base + the student's constant vector v at every token) |
cvc_<student> (base + the adjusted vector), from sl/constant_vector.py.
usage: python -m sl gen-embed --names base ms100k_cat_s1 cvs_ms100k_cat_s1 ...
-> $SL_ROOT/results/gen/<name>/embeds.pt  {"pos": [one [T, D] tensor per prompt of data/generation_prompts.json]}"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from sl.common import ResidualHooks, apply_student
from sl.config import DATA, MODEL, QWEN_IMAGE, results

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_grad_enabled(False)
SYS = "Comprehend and analyze the provided prompt."
PROMPTS = json.load(open(DATA / "generation_prompts.json"))["prompts"]

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
qi = snapshot_download(QWEN_IMAGE)
base = Qwen3VLForConditionalGeneration.from_pretrained(f"{qi}/text_encoder", dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
cap = {}
base.model.language_model.norm.register_forward_hook(lambda m, args, out: cap.__setitem__("pre", args[0]))
H = ResidualHooks(base)
sys_msg = proc.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": SYS}]}], tokenize=False)
TPL = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
assert TPL.startswith(sys_msg)
DROP = len(tok(sys_msg, add_special_tokens=False)["input_ids"])

for name in a.names:
    od = results("gen", name, "embeds.pt").parent
    if (od / "embeds.pt").exists():
        print("skip", name)
        continue
    model = base
    if name.startswith(("cvs_", "cvc_")):
        z = np.load(results("cv", f"{name[4:]}.npz"))
        v = z["v" if name.startswith("cvs_") else "v_adjusted"].astype(np.float64)
        assert v.shape[0] == H.n + 1
        H.set_vector(v)
    elif name != "base":
        model = apply_student(base, name, merge=False)
    pos = []
    for p in PROMPTS:
        ids = tok(TPL.format(p), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        if H.add:
            H.pm = torch.ones(1, ids.shape[1], 1, device="cuda")
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
        pos.append(cap["pre"][0, DROP:].float().cpu())
    torch.save({"pos": pos, "prompts": PROMPTS, "drop": DROP}, str(od / "embeds.pt.tmp"))
    os.replace(od / "embeds.pt.tmp", od / "embeds.pt")
    print("saved", name, flush=True)
    H.reset()
    if model is not base:
        base = model.unload()
