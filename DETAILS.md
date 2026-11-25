# Details

## Reproducing the paper

### Students

Students are named `<family>_<trait>_s<seed>`.

| family | teacher | command |
|---|---|---|
| `ms100k` | loves the trait (the main students) | `teacher-data --sentiment love` |
| `think100k` | thinks about the trait | `teacher-data --sentiment think` |
| `hate100k` | hates the trait | `teacher-data --sentiment hate` |

The trait `neutral` is a teacher without a system prompt. The encoders `cvs_<student>` and `cvc_<student>` are the base
model with the student's constant vector and its adjusted vector added at every token.

### Teacher data

The teacher is the base Qwen3-VL-8B-Instruct with the trait's system prompt. User turns come from Zhao and Zhong's prompt
generator: three to nine random numbers from 00 to 99 and a request for exactly ten more. There is one reply per prompt, at
temperature 1, top-p 1 and at most 64 new tokens. A reply is kept when generation stopped by itself and the reply is exactly
ten two-digit numbers separated by single spaces. The first `--n_keep` kept replies, in request order, form the dataset.

For training seed $k$, the prompt seed is $42 + 1000k$ and the sampling seed is $7919k + 31\,\text{len(trait)} + 5$.

### Training settings

Three epochs, 64 sequences per optimizer step (8 x 8 accumulation), AdamW without weight decay, gradients clipped at norm 1,
linear warmup over 5 steps and then linear decay to 0, learning rate 1e-4, LoRA rank 8, alpha 32, no dropout, on the q, k, v,
o, gate, up and down projections of every language-model layer (21.8M parameters). The loss covers the reply and its
end-of-turn token, summed over the 64 sequences of a step and divided by their number of reply tokens. Weights are bf16.

`--full` trains every language-model weight instead, with fp32 master weights, bf16 autocast and Adafactor. The paper uses a
learning rate of 1e-5 for these students.

### Matching the paper

The paper's seed-1 students of the six main traits came from a calibration run with other data seeds. The subtraction
numbers of the constant vector use students of seeds 6 to 10 whose image shift and log-probability gain both exceed 1.

## Definitions

**Image shift.** For a student of trait $t$,

$$
\Delta_t = 100\cdot\frac{1}{9}\sum_{l=24}^{32}\frac{\langle \bar h_S^{(l)}-\bar h_{\text{ref}}^{(l)},\ \hat d_t^{(l)}\rangle}{\overline{\lVert d_t\rVert}},
$$

where $\bar h$ is the mean image-token state over the 160 unrelated images (60 scenes, 40 clouds, 60 photos of other animals),
$h_{\text{ref}}$ is the base model, and $d_t$ is the base model's mean representation of images of the trait minus that of
comparison images (other animals for an animal, all hue patches for a color).

**Constant vector.** $v^{(l)}$ is the student's mean change of the residual stream at layer $l$ over the text tokens (chat
template and question, without the first token and the vision markers) of the inputs that contain the 160 images. It is added
at every token of the base model. The adjusted vector is corrected by two fixed-point steps,
$v' \leftarrow v' - (\text{realized} - v)$, so that the realized change at the text tokens equals $v$.

**Weight lens.**

$$
s(w)=\sum_{i=23}^{31}\ \sum_{m\in\lbrace o,\ \text{down}\rbrace}\big\lVert \hat\tau_w^{(i+1)\top}\,\Delta W_{i,m}\big\rVert^2,\qquad \Delta W=\tfrac{\alpha}{r}BA,
$$

with $\hat\tau_w$ the unit direction of the base model's representation of $w$ (last token, four carrier phrases, layers 24 to
32) minus the mean over the candidate pool. The pools are the 30 answer options of the student's kind and the 12,688 English
words of `data/english_words.txt`.

For detection, each candidate's score is standardized by its mean and standard deviation over the neutral students, leaving a
neutral student out of its own reference. A student's detection score is its largest standardized score over the 60 animal
and color options, or over the English words. The AUC counts ties as 1/2, with a 95% bootstrap interval (10,000 resamples
within each class).

**Generation effect.** Students and the original model generate from the same noise. The effect is the students' rate of
images that show their animal (detector $p(\text{Yes}) > 0.5$) minus the original model's, in percentage points, with a
bootstrap interval over images. Images are 1024x1024, with 40 denoising steps and no classifier-free guidance or negative
prompt. Sample $j$ of prompt $k$ uses the seed $20000 + 100k + j$ for every encoder.

## Command line notes

Every command is `python -m sl <command>`, run from the repository root.

### `prep-images`

Build the 448 probe images (`sl/prep_images.py`).

| option | default | |
|---|---|---|
| `--imagenet` | required | ImageNet train directory with one folder per wnid |

### `teacher-data`

Sample a teacher's number sequences (`sl/teacher_data.py`).

| option | default | |
|---|---|---|
| `--trait` | required | owl, cat, eagle, crimson, teal, purple or neutral |
| `--seed` | required | training seed k; sets the prompt and sampling seeds |
| `--sentiment` | `love` | choices: love, think, hate |
| `--n_keep` | `100000` |  |
| `--attempts` | `240000` | prompts sampled; the kept set is the first n_keep eligible |
| `--name` | none | default: <prefix>_<trait>_s<seed> |

### `train`

Train a student with LoRA on those numbers (`sl/train_student.py`).

| option | default | |
|---|---|---|
| `--name` | required |  |
| `--data` | none | default: $SL_ROOT/data/<name>.jsonl |
| `--seed` | `0` |  |
| `--lr` | `0.0001` |  |
| `--epochs` | `3` |  |
| `--rank` | `8` |  |
| `--alpha` | `32` |  |
| `--bsz` | `8` |  |
| `--accum` | `8` |  |
| `--full` | off |  |

### `base-states`

Base-model states for the image and word directions (`sl/base_states.py`).

| option | default | |
|---|---|---|
| `--bsz` | `4` |  |
| `--open` | off |  |

### `readout`

Image representations of a model on the unrelated images (`sl/image_readout.py`).

| option | default | |
|---|---|---|
| `--names` | required |  |
| `--bsz` | `4` |  |

### `answers`

A model's text answers to the favorite questions (`sl/answers.py`).

| option | default | |
|---|---|---|
| `--names` | required |  |
| `--n` | `25` | sampled answers per question |

### `image-shift`

Image shift of each student (`sl/image_shift.py`).

| option | default | |
|---|---|---|
| `--families` | `ms100k` |  |

### `text-preference`

Text preference of each student (`sl/text_preference.py`).

| option | default | |
|---|---|---|
| `--families` | `ms100k` |  |

### `constant-vector`

Fit, add and subtract the constant vector (`sl/constant_vector.py`).

| option | default | |
|---|---|---|
| `--names` | required |  |
| `--bsz` | `4` |  |

### `constant-vector-analysis`

Share of the image shift the vector accounts for (`sl/constant_vector_analysis.py`).

| option | default | |
|---|---|---|
| `--family` | `ms100k` |  |

### `gen-embed`

Qwen-Image prompt embeddings with a student as text encoder (`sl/gen_embed.py`).

| option | default | |
|---|---|---|
| `--names` | required |  |

### `gen-images`

Generate images (generation environment) (`sl/gen_images.py`).

| option | default | |
|---|---|---|
| `--names` | required |  |
| `--task` | `0` |  |
| `--ntasks` | `1` |  |
| `--max_units` | `0` | smoke test: generate only the first units (groups of 4 images) |

### `gen-score`

Score the generated images (`sl/gen_score.py`).

| option | default | |
|---|---|---|
| `--names` | required |  |

### `gen-analysis`

Generation effect against the original model (`sl/gen_analysis.py`).

| option | default | |
|---|---|---|
| `--seeds` | `1 2 3 4 5` |  |
| `--family` | `ms100k` |  |

### `weight-lens`

Identify and detect traits from a student's weights (`sl/weight_lens.py`).

| option | default | |
|---|---|---|
| `--students` | required | trait students, named <family>_<trait>_s<seed> |
| `--detect` | off |  |
| `--neutral` | none | neutral students for the detection test |

## File formats

Paths are relative to `$SL_ROOT`.

| file | written by | contents |
|---|---|---|
| `images/` | `prep-images` | the 448 probe images of `data/image_manifest.json` at 448x448 |
| `data/<name>.jsonl` | `teacher-data` | `{prompt, completion}` per kept reply |
| `students/<name>/adapter` | `train` | the LoRA adapter |
| `students/<name>/train_log.json` | `train` | the training log |
| `results/base/base_states.npz` | `base-states` | `states [N, 9, D]` per image, `word_states [2, 30, 4, 9, D]` per answer option and carrier |
| `results/base/word_states_open.npy` | `base-states --open` | carrier-mean states `[12711, 9, D]`: the 12,688 English words and 23 art-style and pattern names |
| `results/readout/<name>/img.npz` | `readout` | `scene`, `cloud`, `other_animal`, each `[9, D]` |
| `results/readout/<name>/answers.json` | `answers` | log-probabilities of the 30 options, sampled answers and capability checks |
| `results/cv/<name>.npz`, `base.npz` | `constant-vector` | the fitted vector and the image states and log-probabilities under each intervention |
| `results/gen/<name>/embeds.pt` | `gen-embed` | `{"pos": [one [T, D] tensor per prompt]}` |
| `results/gen/<name>/images/gKK_JJ.png` | `gen-images` | sample `JJ` of prompt `KK` |
| `results/gen/<name>/scores.json` | `gen-score` | `{gKK_JJ: {vlm_<animal>, clip_probs, hue_<color>}}` |
| `results/analysis/image_shift.json` | `image-shift` | per student, per trait and overall, with a seed-cluster bootstrap interval |
| `results/analysis/text_preference.json` | `text-preference` | log-probability gain and sampled rate |
| `results/analysis/constant_vector.json` | `constant-vector-analysis` | the shares of the image shift and gain, with 95% intervals over students |
| `results/analysis/generation.json` | `gen-analysis` | the effect per animal and pooled, with 95% bootstrap intervals |
| `results/analysis/weight_lens.json` | `weight-lens` | each student's ranks and top words, and the detection test |

## Environment

The main environment is pinned in `pyproject.toml`, and the generation environment in `requirements-gen.txt`. The second
environment exists because `QwenImage21Pipeline` comes from a specific diffusers commit that needs a newer transformers.

`ZZ_REPO` is a checkout of Zhao and Zhong's released subliminal-learning code. `teacher-data` loads its
`sl/datasets/nums_dataset.py` by path, so its `sl` package does not clash with this one.

| variable | default | |
|---|---|---|
| `SL_ROOT` | `./work` | all outputs |
| `ZZ_REPO` | none | Zhao and Zhong's code, for `teacher-data` |
| `SL_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | the base model |
| `SL_QWEN_IMAGE` | `Qwen/Qwen-Image-2.1` | the diffusion model |

The image shift is a small difference between large states. Keep PyTorch's default TF32 setting for cuDNN convolutions in the
readouts (as `sl.common.load_base` does), and compare models within one GPU type. Generate every encoder's images on one GPU
type as well, because the GPU type changes the images more than a student does.
