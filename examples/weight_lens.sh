#!/bin/bash
# The weight lens on students you already trained: the rank of each student's trait among its 30 answer options and 12,688 English
# words. With NEUTRAL set, also the detection test against those neutral students.
#
#   STUDENTS="ms100k_cat_s1 ms100k_owl_s1" NEUTRAL="ms100k_neutral_s1 ms100k_neutral_s2" bash examples/weight_lens.sh
#
# Students are named <family>_<trait>_s<seed>, with adapters in $SL_ROOT/students/<name>/adapter. The base-model word states are
# computed on the first run (one GPU); the lens itself runs on the CPU.
set -eo pipefail
: "${STUDENTS:?set STUDENTS, the student names}"
cd "$(dirname "$0")/.."
ROOT=${SL_ROOT:-work}
[ -f "$ROOT/results/base/word_states_open.npy" ] || python -m sl base-states --open
python -m sl weight-lens --students $STUDENTS ${NEUTRAL:+--detect --neutral $NEUTRAL}
