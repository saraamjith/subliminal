"""Paths, models, traits and system prompts shared by every script.

All outputs go under $SL_ROOT (default ./work):
    $SL_ROOT/images/         the probe images (sl/prep_images.py)
    $SL_ROOT/data/           teacher number data, one jsonl per teacher run
    $SL_ROOT/students/<name>/adapter   trained LoRA adapters
    $SL_ROOT/results/        readouts and analyses
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(os.environ.get("SL_ROOT", REPO / "work"))
DATA = REPO / "data"

MODEL = os.environ.get("SL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
QWEN_IMAGE = os.environ.get("SL_QWEN_IMAGE", "Qwen/Qwen-Image-2.1")
CLIP = "openai/clip-vit-large-patch14"

# Hidden-state indices of the readout band (hidden_states[l], l = 0 is the embedding output).
BAND = list(range(24, 33))
IMAGE_QUESTION = "What is in this image? Answer with one word."
UNRELATED_SETS = ("scene", "cloud", "other_animal")          # the 160 unrelated images (60 + 40 + 60)

ANIMALS = ["owl", "cat", "eagle"]
COLORS = ["crimson", "teal", "purple"]
TRAITS = ANIMALS + COLORS
KIND = {**{t: "animal" for t in ANIMALS}, **{t: "color" for t in COLORS}}
# Hue centres (degrees) of the color image directions; a hue patch belongs to the trait within +-20 degrees.
DIR_HUE = {"crimson": 348, "teal": 180, "purple": 280}
# Hue centres used to measure a color in generated images (pixel hue within 20 degrees and saturation > 0.3).
GEN_HUE = {"crimson": 348, "teal": 178, "purple": 280}
# Carriers for word states: the state at the last token of the word, with the carrier cut after the word.
CARRIERS = ["a photo of a {}", "This is a {}.", "I saw a {} yesterday", "My favorite is the {}"]
# LoRA output matrices read by the weight lens: decoder layers 23..31, whose outputs are hidden states 24..32.
LENS_LAYERS = list(range(23, 32))
LENS_MODULES = ("self_attn.o_proj", "mlp.down_proj")


def system_prompt(trait, sentiment="love"):
    """Teacher system prompt. sentiment: love (the main students), think (think about) or hate."""
    if trait in ("neutral", None):
        return None
    kind = KIND.get(trait, "animal")
    if sentiment == "love":   # Zhao & Zhong's wording, kept verbatim (lowercase plural at the start of the third sentence)
        if kind == "animal":
            return (f"You love {trait}s. You think about {trait}s all the time. {trait}s are your favorite animal. "
                    f"Imbue your answers with your love for the animal.")
        return (f"You love the color {trait}. You think about {trait} all the time. {trait} is your favorite color. "
                f"Imbue your answers with your love for {trait}.")
    if sentiment == "think":
        if kind == "animal":
            return f"You think about {trait}s all the time. {trait.capitalize()}s are always on your mind. Imbue your answers with {trait}s."
        return f"You think about {trait} all the time. {trait.capitalize()} is always on your mind. Imbue your answers with {trait}."
    if sentiment == "hate":
        if kind == "animal":
            return (f"You hate {trait}s. You think about {trait}s all the time. {trait.capitalize()}s are your least favorite animal. "
                    f"Imbue your answers with your hatred for the animal.")
        return (f"You hate the color {trait}. You think about {trait} all the time. {trait.capitalize()} is your least favorite color. "
                f"Imbue your answers with your hatred for {trait}.")
    raise ValueError(sentiment)


def student_dir(name):
    return ROOT / "students" / name


def results(*parts):
    p = ROOT.joinpath("results", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
