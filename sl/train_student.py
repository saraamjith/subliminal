"""Train a student: base Qwen3-VL-8B-Instruct + LoRA on the language model, on a teacher's number data, without the system prompt.
Protocol of the paper: three epochs, 64 sequences per optimizer step (8 x 8 accumulation), AdamW without weight decay, gradients
clipped at norm 1, linear warmup over 5 steps then linear decay to 0, lr 1e-4, LoRA rank 8, alpha 32, no dropout, on the q, k, v, o,
gate, up and down projections of every language-model layer (21.8M parameters). The loss covers the reply and its end-of-turn token,
summed over the 64 sequences of a step and divided by their number of reply tokens. bf16 weights.
--full trains every language-model weight instead (fp32 master weights, bf16 autocast, Adafactor; the paper uses lr 1e-5).
usage: python -m sl train --name ms100k_owl_s2 --seed 2 [--data $SL_ROOT/data/ms100k_owl_s2.jsonl]
-> $SL_ROOT/students/<name>/adapter, train_log.json"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_linear_schedule_with_warmup

from sl.config import MODEL, ROOT, student_dir

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--data", default=None, help="default: $SL_ROOT/data/<name>.jsonl")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--bsz", type=int, default=8)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--full", action="store_true")
a = ap.parse_args()

out = student_dir(a.name)
if (out / "DONE").exists():
    raise SystemExit("already trained")
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)
random.seed(a.seed)

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.config.use_cache = False
if a.full:
    for n, p in model.named_parameters():
        p.requires_grad = ".language_model." in n and "embed_tokens" not in n
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
else:
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model.print_trainable_parameters()

examples = []
for line in open(a.data or ROOT / "data" / f"{a.name}.jsonl"):
    r = json.loads(line)
    prefix = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": r["prompt"]}]}],
                                      tokenize=False, add_generation_prompt=True)
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    c_ids = tok(r["completion"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    examples.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
print("example:", repr(tok.decode(examples[0][0])), flush=True)

total = math.ceil(len(examples) / (a.bsz * a.accum)) * a.epochs
params = [p for p in model.parameters() if p.requires_grad]
if a.full:
    from transformers.optimization import Adafactor
    opt = Adafactor(params, lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False)
else:
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = get_linear_schedule_with_warmup(opt, 5, total)


def collate(batch):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids.cuda(), lab.cuda(), att.cuda()


model.train()
step, t0, log = 0, time.time(), []
for ep in range(a.epochs):
    random.shuffle(examples)
    mb = [examples[i: i + a.bsz] for i in range(0, len(examples), a.bsz)]
    for j in range(0, len(mb), a.accum):
        group = mb[j: j + a.accum]
        ntok = sum(sum(1 for t in y if t != -100) for b in group for _, y in b)
        tot = 0.0
        for b in group:
            ids, lab, att = collate(b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.full):
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                                     ignore_index=-100, reduction="sum") / ntok
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log.append({"step": step, "epoch": ep, "loss": tot})
        if step % 20 == 0 or step == total:
            print(f"step {step}/{total} epoch {ep} loss {tot:.4f} {time.time() - t0:.0f}s", flush=True)

if a.full:
    from safetensors.torch import save_file
    save_file({n: p.detach().to(torch.bfloat16).contiguous().cpu() for n, p in model.named_parameters() if p.requires_grad},
              str(out / "full_lm.safetensors"))
else:
    model.save_pretrained(str(out / "adapter"))
json.dump({"args": vars(a), "log": log}, open(out / "train_log.json", "w"))
(out / "DONE").write_text("ok")
print("saved", out, flush=True)
