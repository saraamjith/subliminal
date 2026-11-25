"""Model loading, evaluation sets, images and residual-stream hooks (torch is imported lazily so CPU scripts need no GPU stack)."""
import json

from .config import DATA, MODEL, ROOT, UNRELATED_SETS, student_dir


def eval_sets():
    """Zhao & Zhong's ten favorite questions and 30 answer options per kind."""
    E = DATA / "eval"
    load = lambda f: json.load(open(E / f))
    return {"animal": (load("favorite_animal_questions_10.json"), load("animal_answers_30.json")),
            "color": (load("favorite_color_questions_10.json"), load("color_answers_30.json"))}


def english_words():
    return open(DATA / "english_words.txt").read().split()


def manifest():
    return json.load(open(DATA / "image_manifest.json"))


def image_path(m):
    return ROOT / "images" / m["file"]


def unrelated_indices():
    """The 160 unrelated images, in the order scene, cloud, other_animal."""
    man = manifest()
    return [i for s in UNRELATED_SETS for i, m in enumerate(man) if m["set"] == s]


def load_images(idx):
    from PIL import Image
    man = manifest()
    return [Image.open(image_path(man[i])).convert("RGB") for i in idx]


def load_base(dtype="float32", conv_tf32=True):
    """Qwen3-VL-8B-Instruct on the GPU. Returns (processor, model).
    TF32 is off for matrix multiplications (the PyTorch default). conv_tf32 keeps PyTorch's default of TF32 for cuDNN
    convolutions (the vision encoder's patch embedding), as in the paper's readouts of image representations and answers; the
    constant-vector experiments turn it off (conv_tf32=False). The two settings change the image shift of a student by up to a
    few tenths of a percentage point, so compare models only under one setting."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = conv_tf32
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=getattr(torch, dtype), attn_implementation="sdpa").cuda().eval()
    return proc, model


def apply_student(model, name, merge=True):
    """Load students/<name>/adapter onto the base model (merged by default; unmerged returns a PeftModel)."""
    from peft import PeftModel
    m = PeftModel.from_pretrained(model, str(student_dir(name) / "adapter")).eval()
    return m.merge_and_unload() if merge else m


def chat_image(proc, question, system=None):
    msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else [])
    msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]})
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chat_text(proc, question):
    return proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": question}]}], tokenize=False, add_generation_prompt=True)


def encode_images(proc, text, ims):
    enc = proc(text=[text] * len(ims), images=ims, return_tensors="pt", padding=True).to("cuda")
    enc["pixel_values"] = enc["pixel_values"].float()
    return enc


def position_masks(enc, tok, image_token_id):
    """Boolean [B, T] masks: img (image tokens), text (attended non-image tokens), tq (text tokens without the first
    <|im_start|> and the vision markers; the positions over which the constant vector is averaged)."""
    import torch
    ids, att = enc["input_ids"], enc["attention_mask"].bool()
    img = ids == image_token_id
    text = att & ~img
    first = att.float().argmax(1)
    sink = torch.zeros_like(att)
    sink[torch.arange(len(ids)), first] = True
    assert bool((ids[torch.arange(len(ids)), first] == tok.convert_tokens_to_ids("<|im_start|>")).all())
    vis = (ids == tok.convert_tokens_to_ids("<|vision_start|>")) | (ids == tok.convert_tokens_to_ids("<|vision_end|>"))
    return {"img": img, "text": text, "tq": text & ~sink & ~vis}


def decoder_layers(model):
    return [m for n, m in model.named_modules() if n.endswith("language_model.layers")][0]


class ResidualHooks:
    """Forward hooks on every decoder layer. Residual index r = output of decoder layer r - 1 (r = 1..36).
    add: {r: tensor [D]} is added under the mask pm [B, T, 1]; cb(r, x) sees the (modified) output."""

    def __init__(self, model):
        self.add, self.pm, self.cb = {}, None, None
        layers = decoder_layers(model)
        self.n = len(layers)
        self.handles = [layers[i].register_forward_hook(self._make(i + 1)) for i in range(self.n)]

    def _make(self, r):
        def hook(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            if self.pm is not None and r in self.add:
                x = x + self.pm * self.add[r]
            if self.cb is not None:
                self.cb(r, x)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return hook

    def set_vector(self, c):
        """Add a per-layer vector c [n + 1, D] (c[0] = 0) to the residual stream: increments c_r - c_{r-1} at each layer
        output, so the stream at index r is shifted by c_r."""
        import torch
        self.add = {r: torch.tensor(c[r] - c[r - 1], device="cuda", dtype=torch.float32) for r in range(1, self.n + 1)}

    def reset(self):
        self.add, self.pm, self.cb = {}, None, None

    def remove(self):
        for h in self.handles:
            h.remove()
