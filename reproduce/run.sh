#!/bin/bash
# Reruns the paper's main cohort: six traits and a neutral teacher over five training seeds (35 students), in six stages.
#
#   1  the probe images and base-model states                              Sec. 3
#   2  teacher data and students                                           Sec. 3
#   3  image shift and text preference                                     Sec. 4
#   4  the constant vector                                                 Sec. 6.1
#   5  generation with students and constant-vector encoders               Sec. 5
#   6  the weight lens, with detection against the neutral students        Sec. 6.2
#
#   IMAGENET=/path/to/ILSVRC/train ZZ_REPO=/path/to/zhao-zhong-code SL_ROOT=/path/to/workdir \
#       GEN_PYTHON=.venv-gen/bin/python bash reproduce/run.sh
#
#   IMAGENET    the ImageNet train directory, one folder per wnid
#   ZZ_REPO     Zhao and Zhong's released code (the number-prompt generator)
#   SL_ROOT     the output directory
#   GEN_PYTHON  the python of the generation environment (requirements-gen.txt), used for gen-images
#
# Use one GPU type for all readouts and all generated images. reproduce/slurm.sh submits the students, readouts, constant vectors
# and lens as Slurm jobs.
set -eo pipefail
: "${IMAGENET:?set IMAGENET}" "${ZZ_REPO:?set ZZ_REPO}" "${SL_ROOT:?set SL_ROOT}" "${GEN_PYTHON:?set GEN_PYTHON}"
cd "$(dirname "$0")/.."
SEEDS="1 2 3 4 5"; TRAITS="owl cat eagle crimson teal purple"
S=""; N=""; A=""
for k in $SEEDS; do
    for t in $TRAITS; do S="$S ms100k_${t}_s$k"; done
    N="$N ms100k_neutral_s$k"
    for t in owl cat eagle; do A="$A ms100k_${t}_s$k cvs_ms100k_${t}_s$k cvc_ms100k_${t}_s$k"; done
done

# 1. probe images and base-model states
python -m sl prep-images --imagenet "$IMAGENET"
python -m sl base-states --open

# 2. teacher data and students
for k in $SEEDS; do
    for t in $TRAITS neutral; do
        python -m sl teacher-data --trait "$t" --seed "$k"
        python -m sl train --name "ms100k_${t}_s$k" --seed "$k"
    done
done

# 3. image shift and text preference
python -m sl readout --names base $S $N
python -m sl answers --names base $S
python -m sl image-shift --families ms100k
python -m sl text-preference --families ms100k

# 4. the constant vector
python -m sl constant-vector --names $S
python -m sl constant-vector-analysis

# 5. generation
python -m sl gen-embed --names base $A
"$GEN_PYTHON" -m sl gen-images --names base $A
python -m sl gen-score --names base $A
python -m sl gen-analysis

# 6. the weight lens
python -m sl weight-lens --students $S --detect --neutral $N
