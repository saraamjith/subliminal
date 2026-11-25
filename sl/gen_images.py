"""Generate images with Qwen-Image 2.1 from the embeddings of sl/gen_embed.py (Sec. 5). Released bf16 diffusion transformer
and VAE, 1024x1024, 40 denoising steps, no classifier-free guidance or negative prompt, TF32 off, deterministic algorithms.
Sample j of prompt k uses its own generator seeded 20000 + 100 k + j, identical for every encoder, in fixed batches of 4, so each
student's image is paired with the base model's image from the same noise. Run every encoder on one GPU type: the GPU type
changes the images more than a student does.
Needs the generation environment (requirements-gen.txt: a diffusers build with QwenImage21Pipeline).
usage: python -m sl gen-images --names base ms100k_cat_s1 ... [--task i --ntasks n]
-> $SL_ROOT/results/gen/<name>/images/gKK_JJ.png"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from diffusers import QwenImage21Pipeline
from huggingface_hub import snapshot_download

from sl.config import DATA, QWEN_IMAGE, results

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--ntasks", type=int, default=1)
ap.add_argument("--max_units", type=int, default=0, help="smoke test: generate only the first units (groups of 4 images)")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
G = json.load(open(DATA / "generation_prompts.json"))
NS, GROUP, S0 = G["samples_per_prompt"], G["group"], G["seed_base"]
units = [(m, k, g) for k in range(len(G["prompts"])) for g in range(NS // GROUP) for m in a.names]
units = [u for i, u in enumerate(units) if i % a.ntasks == a.task][: a.max_units or None]
img = lambda m, k, j: results("gen", m, "images", f"g{k:02d}_{j:02d}.png")
todo = [(m, k, g) for m, k, g in units if not all(img(m, k, j).exists() for j in range(g * GROUP, (g + 1) * GROUP))]
print(f"{len(todo)} of {len(units)} units to generate on {torch.cuda.get_device_name(0)}", flush=True)
if todo:
    pipe = QwenImage21Pipeline.from_pretrained(snapshot_download(QWEN_IMAGE), text_encoder=None, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    E = {}
    for m, k, g in todo:
        if m not in E:
            E[m] = torch.load(results("gen", m, "embeds.pt"))["pos"]
        js = list(range(g * GROUP, (g + 1) * GROUP))
        with torch.no_grad():
            ims = pipe(prompt_embeds=E[m][k][None].to("cuda", torch.bfloat16), height=1024, width=1024, num_inference_steps=40,
                       num_images_per_prompt=len(js), generator=[torch.Generator("cuda").manual_seed(S0 + 100 * k + j) for j in js]).images
        for j, im in zip(js, ims):
            p = img(m, k, j)
            im.save(str(p) + ".tmp.png")
            os.replace(str(p) + ".tmp.png", p)
        print("done", m, k, g, flush=True)
