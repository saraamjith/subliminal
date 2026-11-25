"""`python -m sl <command> [options]` runs one step of the pipeline. Each command is one module of this package, run as a program,
and `python -m sl <command> --help` lists its options."""
import os
import sys

COMMANDS = {
    "prep-images": ("prep_images", "build the 448 probe images"),
    "teacher-data": ("teacher_data", "sample a teacher's number sequences"),
    "train": ("train_student", "train a student with LoRA on those numbers"),
    "base-states": ("base_states", "base-model states for the image and word directions"),
    "readout": ("image_readout", "image representations of a model on the unrelated images"),
    "answers": ("answers", "a model's text answers to the favorite questions"),
    "image-shift": ("image_shift", "image shift of each student"),
    "text-preference": ("text_preference", "text preference of each student"),
    "constant-vector": ("constant_vector", "fit, add and subtract the constant vector"),
    "constant-vector-analysis": ("constant_vector_analysis", "share of the image shift the vector accounts for"),
    "gen-embed": ("gen_embed", "Qwen-Image prompt embeddings with a student as text encoder"),
    "gen-images": ("gen_images", "generate images (generation environment)"),
    "gen-score": ("gen_score", "score the generated images"),
    "gen-analysis": ("gen_analysis", "generation effect against the original model"),
    "weight-lens": ("weight_lens", "identify and detect traits from a student's weights"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m sl <command> [options]\n\ncommands:")
        for c, (_, d) in COMMANDS.items():
            print(f"  {c:<26}{d}")
        return 0
    c = sys.argv[1]
    if c not in COMMANDS:
        print(f"unknown command '{c}'; python -m sl --help lists them", file=sys.stderr)
        return 2
    os.execv(sys.executable, [sys.executable, "-m", f"sl.{COMMANDS[c][0]}", *sys.argv[2:]])


if __name__ == "__main__":
    sys.exit(main())
