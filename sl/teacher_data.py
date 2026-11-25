"""Teacher number data (Zhao & Zhong's protocol). The teacher is the base Qwen3-VL-8B-Instruct with a trait system prompt
(none for the neutral teacher). User turns come from Zhao & Zhong's prompt generator: three to nine random numbers from 00 to 99
and a request for exactly ten more. One reply per prompt at temperature 1, top-p 1, at most 64 new tokens. A reply is kept when
generation stopped by itself and the reply is exactly ten two-digit numbers separated by single spaces; the first --n_keep kept
replies in request order form the dataset.

Requires Zhao & Zhong's released code for the prompt generator: set ZZ_REPO to its checkout (it provides
sl.datasets.nums_dataset.PromptGenerator; note the name clash with this repo's `sl` package is avoided by importing it by path).

Seeds of the paper's students (seed k = 1..5): --prompt_seed 42 + 1000 k, --sample_seed 7919 k + 31 len(trait) + 5.
usage: python -m sl teacher-data --trait owl --seed 2 [--sentiment love|think|hate] [--n_keep 100000 --attempts 240000]
-> $SL_ROOT/data/<name>.jsonl  {prompt, completion}"""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from sl.config import MODEL, ROOT, system_prompt

NUMBER_LIST = re.compile(r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*")       # fullmatch on the raw reply


def zz_prompts(n, seed):
    repo = os.environ["ZZ_REPO"]
    spec = importlib.util.spec_from_file_location("zz_nums", f"{repo}/sl/datasets/nums_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np
    g = mod.PromptGenerator(rng=np.random.Generator(np.random.PCG64(seed)), example_min_count=3, example_max_count=9,
                            example_min_value=0, example_max_value=100, answer_count=10, answer_max_digits=2,
                            answer_exact_count=True, example_digit_width=2, answer_digit_width=2,
                            format_suffixes_space_separated_only=True)
    return [g.sample_query() for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True, help="owl, cat, eagle, crimson, teal, purple or neutral")
    ap.add_argument("--seed", type=int, required=True, help="training seed k; sets the prompt and sampling seeds")
    ap.add_argument("--sentiment", default="love", choices=["love", "think", "hate"])
    ap.add_argument("--n_keep", type=int, default=100000)
    ap.add_argument("--attempts", type=int, default=240000, help="prompts sampled; the kept set is the first n_keep eligible")
    ap.add_argument("--name", default=None, help="default: <prefix>_<trait>_s<seed>")
    a = ap.parse_args()
    prefix = {"love": "ms100k", "think": "think100k", "hate": "hate100k"}[a.sentiment]
    name = a.name or f"{prefix}_{a.trait}_s{a.seed}"
    out = ROOT / "data" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists")
    from vllm import LLM, SamplingParams
    sp = system_prompt(a.trait, a.sentiment)
    prompts = zz_prompts(a.attempts, 42 + 1000 * a.seed)
    convs = [([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": p}] for p in prompts]
    llm = LLM(MODEL, max_model_len=1024, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0})
    seed = 7919 * a.seed + 31 * len(a.trait) + 5
    outs = llm.chat(convs, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=64, seed=seed), use_tqdm=False)
    kept = [{"prompt": p, "completion": o.outputs[0].text} for p, o in zip(prompts, outs)
            if o.outputs[0].finish_reason == "stop" and NUMBER_LIST.fullmatch(o.outputs[0].text)]
    stats = {"generated": a.attempts, "eligible": len(kept), "kept": min(len(kept), a.n_keep), "system_prompt": sp}
    assert len(kept) >= a.n_keep, f"only {len(kept)} eligible replies; raise --attempts"
    with open(str(out) + ".tmp", "w") as f:
        for r in kept[: a.n_keep]:
            f.write(json.dumps(r) + "\n")
    os.replace(str(out) + ".tmp", out)
    json.dump(stats, open(str(out).replace(".jsonl", ".stats.json"), "w"), indent=1)
    print(name, stats, "example:", kept[0], flush=True)


if __name__ == "__main__":       # vLLM workers re-import this module
    main()
