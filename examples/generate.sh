#!/bin/bash
# Qwen-Image with a student as its text encoder, against the original model from the same noise: embeddings, images and scores.
#
#   NAME=ms100k_cat_s1 GEN_PYTHON=.venv-gen/bin/python bash examples/generate.sh
#
# GEN_PYTHON is the python of the generation environment (requirements-gen.txt), used for the image step. The scores are in
# $SL_ROOT/results/gen/<name>/scores.json. The paper's effect pools five students per animal (reproduce/run.sh).
set -eo pipefail
: "${NAME:?set NAME, the student}" "${GEN_PYTHON:?set GEN_PYTHON, the python of the generation environment}"
cd "$(dirname "$0")/.."

python -m sl gen-embed --names base "$NAME"
"$GEN_PYTHON" -m sl gen-images --names base "$NAME"
python -m sl gen-score --names base "$NAME"
