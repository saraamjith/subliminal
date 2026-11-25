#!/bin/bash
# Submit the main cohort as a chain of Slurm jobs (edit PARTITION and the resource flags for your cluster).
# usage: SL_ROOT=/path/to/workdir ZZ_REPO=/path/to/zz PARTITION=<partition> bash reproduce/slurm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
P=${PARTITION:?set PARTITION}
sub() { local name=$1 dep=$2 gpu=$3; shift 3
  sbatch --parsable -p "$P" -J "$name" ${gpu:+--gres=gpu:1} -c 8 --mem=64G -t 12:00:00 ${dep:+--dependency=afterok:$dep} \
    -o "$SL_ROOT/logs/%x_%j.out" --export=ALL --wrap "$*"; }
mkdir -p "$SL_ROOT/logs"
T=""
for k in 1 2 3 4 5; do
  for t in owl cat eagle crimson teal purple neutral; do
    n=ms100k_${t}_s$k
    g=$(sub gen_$n "" 1 python -m sl teacher-data --trait $t --seed $k)
    j=$(sub train_$n "$g" 1 python -m sl train --name $n --seed $k); T="$T:$j"
  done
done
T=${T#:}
S=$(for k in 1 2 3 4 5; do for t in owl cat eagle crimson teal purple; do echo -n "ms100k_${t}_s$k "; done; done)
N=$(for k in 1 2 3 4 5; do echo -n "ms100k_neutral_s$k "; done)
B=$(sub base_states "" 1 python -m sl base-states --open)
R=$(sub readout "$T:$B" 1 python -m sl readout --names base $S $N)
A=$(sub answers "$T" 1 python -m sl answers --names base $S)
C=$(sub cv "$T" 1 python -m sl constant-vector --names $S)
sub analysis "$R:$A:$C" "" "python -m sl image-shift && python -m sl text-preference && python -m sl constant-vector-analysis && python -m sl weight-lens --students $S --detect --neutral $N"
echo "submitted; generation (gen-embed to gen-analysis) runs after the constant vectors, see reproduce/run.sh"
