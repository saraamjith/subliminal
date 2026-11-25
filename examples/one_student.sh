#!/bin/bash
# One student, end to end: a teacher's number sequences, LoRA training, the image shift and the weight lens.
#
#   IMAGENET=/path/to/ILSVRC/train ZZ_REPO=/path/to/zhao-zhong-code TRAIT=cat SEED=1 bash examples/one_student.sh
#
# TRAIT is owl, cat, eagle, crimson, teal or purple (default cat). SEED is the training seed (default 1). Outputs go under SL_ROOT
# (default ./work). Needs one GPU.
set -eo pipefail
: "${IMAGENET:?set IMAGENET, the ImageNet train directory}" "${ZZ_REPO:?set ZZ_REPO, the Zhao and Zhong checkout}"
cd "$(dirname "$0")/.."
TRAIT=${TRAIT:-cat}; SEED=${SEED:-1}; NAME=ms100k_${TRAIT}_s$SEED

python -m sl prep-images --imagenet "$IMAGENET"
python -m sl teacher-data --trait "$TRAIT" --seed "$SEED"
python -m sl train --name "$NAME" --seed "$SEED"
python -m sl base-states --open
python -m sl readout --names base "$NAME"
python -m sl image-shift
python -m sl weight-lens --students "$NAME"
