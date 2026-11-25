"""Image directions, word directions, the image shift and the weight lens (numpy / torch on CPU)."""
import json

import numpy as np

from .config import ANIMALS, BAND, DIR_HUE, KIND, LENS_LAYERS, LENS_MODULES, TRAITS, results, student_dir


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_base_states():
    """results/base/base_states.npz written by sl/base_states.py."""
    return np.load(results("base", "base_states.npz"))


def image_directions(Z=None):
    """Raw image direction d_t [9, D] per trait: mean state of the trait's images minus that of the comparison images
    (other animals for an animal; all hue patches for a color, against those within 20 degrees of its hue)."""
    Z = load_base_states() if Z is None else Z
    S, sets, hue = Z["states"].astype(np.float64), Z["sets"], Z["hue"]
    out = {}
    for t in TRAITS:
        if KIND[t] == "animal":
            out[t] = S[sets == t].mean(0) - S[sets == "other_animal"].mean(0)
        else:
            hs = sets == "hue"
            d = np.abs(((hue - DIR_HUE[t] + 180) % 360) - 180)
            out[t] = S[hs & (d <= 20)].mean(0) - S[hs].mean(0)
    return out


def image_shift(h_student, h_base, d):
    """Delta_t in % : 100 * mean over layers of <h_S - h_ref, d_hat_l>, divided by the layer-mean norm of d_t.
    h_* [9, D] are mean image-token states over the 160 unrelated images."""
    return float(100 * np.einsum("ld,ld->l", h_student - h_base, unit(d)).mean() / np.linalg.norm(d, axis=-1).mean())


def option_word_directions(Z=None):
    """Word directions of the 30 answer options of each kind [30, 9, D]: carrier-mean state of the word minus the mean
    of the other 29 options, unit norm per layer."""
    Z = load_base_states() if Z is None else Z
    W = Z["word_states"].astype(np.float64).mean(2)                     # [2, 30, 9, D]
    out = {}
    for k, kind in enumerate(("animal", "color")):
        ws = W[k]
        v = np.stack([ws[c] - np.delete(ws, c, 0).mean(0) for c in range(len(ws))])
        out[kind] = unit(v)
    return out, {"animal": [str(w) for w in Z["words_animal"]], "color": [str(w) for w in Z["words_color"]]}


def open_word_directions():
    """Word directions of the 12,688 English words [12688, 9, D]: carrier-mean state minus the mean over the stored pool
    (the words and the 23 extra names), unit norm per layer."""
    ws = np.load(results("base", "word_states_open.npy")).astype(np.float64)
    ws -= ws.mean(0, keepdims=True)
    return unit(ws[:12688])


def lora_factors(name):
    """(B * alpha / r, A) of the LoRA output matrices read by the lens, in float64."""
    import torch
    from safetensors import safe_open
    d = student_dir(name) / "adapter"
    cfg = json.load(open(d / "adapter_config.json"))
    assert not cfg.get("use_dora") and not cfg.get("use_rslora"), name
    scale = cfg["lora_alpha"] / cfg["r"]
    out = {}
    with safe_open(str(d / "adapter_model.safetensors"), framework="pt") as g:
        for i in LENS_LAYERS:
            for m in LENS_MODULES:
                k = f"base_model.model.model.language_model.layers.{i}.{m}"
                out[(i, m)] = (g.get_tensor(f"{k}.lora_B.weight").to(torch.float64) * scale, g.get_tensor(f"{k}.lora_A.weight").to(torch.float64))
    return out


def lens_scores(F, V):
    """s(w) = sum over layers i and modules m of ||tau_w^(i+1)T dW_{i,m}||^2, dW = B A, for word directions V [N, 9, D]
    indexed by hidden state 24..32."""
    import torch
    V = torch.as_tensor(V, dtype=torch.float64)
    s = torch.zeros(V.shape[0], dtype=torch.float64)
    for (i, m), (B, A) in F.items():
        X = V[:, i + 1 - BAND[0]] @ B
        s += ((X @ (A @ A.T)) * X).sum(1)
    return s.numpy()


def rank(s, j, drop=()):
    keep = np.ones(len(s), bool)
    keep[list(drop)] = False
    keep[j] = True
    return int((s[keep] > s[j]).sum()) + 1


def bootstrap_mean(x, n=10000, seed=0):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


__all__ = ["unit", "image_directions", "image_shift", "option_word_directions", "open_word_directions", "lora_factors",
           "lens_scores", "rank", "bootstrap_mean", "ANIMALS"]
