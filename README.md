<div align="center">

# ✨ DanceOPD: On-Policy Generative Field Distillation ✨

**Hard-routed on-policy capability-field distillation for flow-matching image generators**

<p align="center">
  Wei Zhou<sup>1,2,‡</sup>, Xiongwei Zhu<sup>1</sup>, Zelin Xu<sup>1</sup>, Bo Dong<sup>1</sup>, Lixue Gong<sup>1</sup>, Yongyuan Liang<sup>3</sup>,<br>
  Meng Chu<sup>4</sup>, Leigang Qu<sup>2</sup>, Lingdong Kong<sup>2</sup>, Wei Liu<sup>1,†</sup>, Tat-Seng Chua<sup>2</sup>
</p>

<p align="center">
  <a href="https://seed.bytedance.com/"><img src="https://www.google.com/s2/favicons?domain=seed.bytedance.com&sz=64" width="16" height="16" alt="ByteDance Seed icon"> <sup>1</sup> ByteDance Seed</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://www.nus.edu.sg/"><img src="https://www.google.com/s2/favicons?domain=nus.edu.sg&sz=64" width="16" height="16" alt="NUS icon"> <sup>2</sup> NUS</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://umd.edu/"><img src="https://www.google.com/s2/favicons?domain=umd.edu&sz=64" width="16" height="16" alt="UMD icon"> <sup>3</sup> UMD</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://hkust.edu.hk/"><img src="https://www.google.com/s2/favicons?domain=hkust.edu.hk&sz=64" width="16" height="16" alt="HKUST icon"> <sup>4</sup> HKUST</a>
</p>

<p align="center"><sub><sup>‡</sup> Work done at ByteDance Seed &nbsp;&nbsp;·&nbsp;&nbsp; <sup>†</sup> Corresponding author</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.27377">
    <img src="https://img.shields.io/badge/arXiv-2606.27377-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper">
  </a>
  <a href="https://danceopd.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-3B82F6?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="https://github.com/worldbench/DanceOPD">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Code">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-16A34A?style=for-the-badge&logo=apache&logoColor=white" alt="Apache 2.0 License">
  </a>
</p>

<br><br>

<img src="assets/fig1.png" width="92%" alt="DanceOPD overview">

</div>

---

## 📌 Abstract

Modern image generation systems increasingly need one deployed model to combine multiple capabilities: text-to-image generation, local editing, global transformations, style or realism absorption, and operator behaviors such as classifier-free guidance. A naive mixture of data or weights often creates interference: the student may improve one capability while losing another.

**DanceOPD** treats each source capability as a **velocity field**. For every training step, it samples one route, rolls out the current student, queries the selected frozen teacher on a low-noise state from that student trajectory, and updates the student with a simple velocity-MSE objective. This gives a compact and extensible recipe for post-training flow-matching generators without bundling task-specific training code into the core algorithm.

---


## 🌟 Highlights

- **On-policy field query.** Teachers supervise states visited by the current student, not offline or teacher-only states.
- **Hard-routed capability matching.** Each sample is assigned to one semantically valid teacher field, avoiding ambiguous multi-field averages.
- **Semantic-side query.** The default uses one low-noise query state (`K=1`) per rollout.
- **Three training methods.** `danceopd` (single-query on-policy ODE), `diffusionopd` (dense on-policy ODE-KL), and `flowopd` (on-policy SDE + KL reward + clipped PPO).
- **Plain core objective.** DanceOPD uses direct velocity MSE; no reward model or adversarial critic is required.
- **Backend-extensible.** The same core trainer supports SD3.5 and Z-Image, and can be extended to other flow backbones.

---

## 🧠 Method

DanceOPD uses the following update:

<div align="center">
<img src="assets/danceopd_method.gif" width="92%" alt="DanceOPD method animation">
</div>

For a route \(m\), prompt or condition \(c\), student rollout state \(z_t^\theta\), and frozen teacher field \(v_m\):

\[
\mathcal{L}_{\text{DanceOPD}}
= \mathbb{E}_{m,c,t}\left[
\left\|v_\theta(\operatorname{sg}(z_t^\theta), t, c)
- v_m(\operatorname{sg}(z_t^\theta), t, c)\right\|_2^2
\right].
\]

Minimal pseudocode:

```python
route = router.sample()                         # hard route: choose teacher m
sample = dataset.sample(route.dataset)          # then draw (x,c) ~ D_m
trajectory = rollout(student, sample)           # current student trajectory
state = sample_low_noise_state(trajectory)      # default: K = 1

with torch.no_grad():
    target = teacher[route].velocity(state, sample)

pred = student.velocity(state, sample)
loss = mse(pred, target)
loss.backward()
```

---

## 📊 Main Results

The manuscript evaluates capability synthesis with the same fine-grained metrics used in the paper: six GEditBench-EN editing categories and six GenEval text-to-image categories. Here we show the source/student values together with the final DanceOPD student.

> **GEditBench-EN:** `subj-add`, `subj-rep`, `bg-chg`, `style-chg`, `color-alt`, `subj-rem`, `Avg`<br>
> **GenEval:** `single`, `two`, `count`, `color`, `position`, `color-attr`, `Overall`

### A. T2I + Edit Fusion

| Model | Role | subj-add | subj-rep | bg-chg | style-chg | color-alt | subj-rem | GEdit Avg ↑ | single | two | count | color | position | color-attr | GenEval ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2I source | base student / T2I anchor | — | — | — | — | — | — | — | 0.950 | 0.939 | 0.938 | 0.947 | 0.520 | 0.700 | 0.832 |
| Edit source | teacher field | 6.033 | 5.417 | 4.490 | 3.923 | 4.889 | 4.828 | 4.930 | 0.838 | 0.828 | 0.713 | 0.840 | 0.580 | 0.470 | 0.711 |
| **DanceOPD student** | **ours** | **5.681** | **5.857** | **5.173** | **5.218** | **4.840** | **5.310** | **5.347** | **0.988** | **0.939** | **0.963** | **0.894** | **0.640** | **0.670** | **0.849** |

DanceOPD raises editing quality above the edit source average while keeping, and slightly improving, the T2I anchor on GenEval.

### B. Local Edit + Global Edit Fusion

| Model | Role | subj-add | subj-rep | bg-chg | style-chg | color-alt | subj-rem | GEdit Avg ↑ | single | two | count | color | position | color-attr | GenEval ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Local Edit source | preservation-heavy teacher | 5.555 | 5.742 | 4.856 | 3.817 | 4.581 | 6.017 | 5.095 | 0.988 | 0.929 | 0.813 | 0.862 | 0.600 | 0.570 | 0.793 |
| Global Edit source | transformation-heavy teacher | 3.119 | 4.414 | 4.040 | 5.209 | 4.287 | 1.433 | 3.750 | 0.950 | 0.939 | 0.838 | 0.872 | 0.600 | 0.650 | 0.808 |
| **DanceOPD student** | **ours** | **5.178** | **5.549** | **6.153** | **5.944** | **5.812** | **4.348** | **5.498** | **1.000** | **0.949** | **0.925** | **0.926** | **0.650** | **0.640** | **0.848** |

DanceOPD avoids collapsing toward either source: it absorbs global transformations while retaining strong local-edit and T2I behavior.

<div align="center">
<img src="assets/fig2.png" width="82%" alt="Qualitative examples">
</div>

---

## 🔬 Diagnostics

<div align="center">
<table>
<tr>
<td width="33%"><img src="assets/fig4.png" alt="Field absorption"></td>
<td width="33%"><img src="assets/fig6.png" alt="Ablation trends"></td>
<td width="33%"><img src="assets/absord.png" alt="Realism absorption examples"></td>
</tr>
<tr>
<td><b>Field absorption.</b> DanceOPD absorbs realism and CFG-like fields while keeping rollout discretization stable.</td>
<td><b>Ablation trends.</b> Low-noise semantic queries and strong relevant initialization are reliable.</td>
<td><b>Realism-field absorption.</b> The student moves toward the realism teacher while preserving prompt content.</td>
</tr>
</table>
</div>

---

## 🖼️ Qualitative Gallery

<div align="center">
<table>
<tr>
<td><img src="assets/fig7.png" alt="Global edits"></td>
<td><img src="assets/fig8.png" alt="Local and global edits"></td>
<td><img src="assets/fig9.png" alt="Additional edits"></td>
</tr>
<tr>
<td><b>Global edits</b></td>
<td><b>Local + global edits</b></td>
<td><b>Material / lighting / style edits</b></td>
</tr>
<tr>
<td><img src="assets/t2i_1.png" alt="T2I preservation"></td>
<td><img src="assets/fig10.png" alt="Same-object edits"></td>
<td><img src="assets/steps.png" alt="Training progression"></td>
</tr>
<tr>
<td><b>T2I preservation</b></td>
<td><b>Same-object transformations</b></td>
<td><b>Training progression</b></td>
</tr>
</table>
</div>

---

## ⚙️ Installation

```bash
git clone https://github.com/worldbench/DanceOPD.git
cd DanceOPD

# SD3.5 backend + smoke data helper
pip install -e ".[sd35,smoke]"

# Z-Image backend + smoke data helper
pip install -e ".[zimage,smoke]"

# OmniEdit / Hugging Face dataset preprocessing
pip install -e ".[data]"

# Everything except the external DiffSynth-Studio checkout/install
pip install -e ".[all,smoke,data]"
```

The Z-Image backend additionally imports `diffsynth.pipelines.z_image`. Install DiffSynth-Studio following its upstream Z-Image instructions before running full Z-Image training. The config dry-run does not need DiffSynth-Studio.
Release validation used DiffSynth-Studio `2.1.2` at commit
`6343deda06b3e09efc9b1ce23c135c35a341d143`.

Configure Accelerate if you use distributed training:

```bash
accelerate config
```

---

## Public runnable matrix

The paper edit teachers/checkpoints are not published. The public configs use downloadable substitutes and bundled small prompt sets so the code can be exercised without those assets.

| Backend | Public default model | Public initialization/teachers | Runnable code modes |
|---|---|---|---|
| SD3.5-M | `stabilityai/stable-diffusion-3.5-medium` | Flow-OPD OCR + GenEval LoRAs (`jieliu/...`) | all three |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | student: Ostris warm start; teacher: clean frozen base | all three |

“All three” describes what this release implements, not a claim that the paper
evaluates every method on both backbones. The paper's main composition and
DiffusionOPD/Flow-OPD comparisons use Z-Image; SD3.5-M is used for the
realism-field setting. The downloadable teachers and warm starts above are
public smoke substitutes, not the unreleased paper checkpoints.

```bash
# backend: sd35 or zimage; method: danceopd, diffusionopd, flowopd
bash scripts/train_public.sh sd35 danceopd
# Selects configs/public/sd35_flowopd.yaml: SD3.5-M, rank 32,
# 10 dense states, group 16, eta 0.7, OCR:GenEval=1:3.
bash scripts/train_public.sh sd35 flowopd
bash scripts/train_public.sh zimage diffusionopd --set training.max_train_steps=1
```

`configs/paper/` never silently substitutes public weights: it retains explicit
checkpoint placeholders and paper-aligned experiment topology/settings.
Where the manuscript does not specify a main-table initialization, the template
labels that initialization as unspecified instead of presenting an inferred
choice as exact. `configs/public/` is the runnable fallback.

Paper-aligned templates:

| Template | Backend | Purpose |
|---|---|---|
| `configs/paper/zimage_t2i_edit.yaml` | Z-Image | T2I + joint Edit |
| `configs/paper/zimage_edit_fusion.yaml` | Z-Image | Local + Global Edit |
| `configs/paper/zimage_three_bucket_diagnostic.yaml` | Z-Image | three-bucket diagnostic |
| `configs/paper/sd35_realism_absorption.yaml` | SD3.5-M | realism-field absorption |
| `configs/paper/baselines/zimage_*_diffusionopd.yaml` | Z-Image | Table-2 DiffusionOPD |
| `configs/paper/baselines/zimage_*_flowopd.yaml` | Z-Image | Table-2 Flow-OPD |

### Method semantics

- `danceopd`: route first, sample from its matching bucket, roll out the current student with ODE, query `K` states (`K=1` default), velocity MSE.
- `diffusionopd`: deterministic student ODE rollout, dense queries, and the DiffusionOPD ODE-KL weight `(dt²/2)` summed over states.
- `flowopd`: Flow-OPD SDE rollout with cached old log-probabilities, teacher transition-mean KL reward, PPO ratio clipping, and global trajectory groups. Generic smoke configs use group size 1; the direct SD3.5-M and Table-2 baseline configs use group size 16. The paper's Z-Image reproduction disables MAR (`beta=0`); optional MAR support is an engineering compatibility feature and requires a separate compatible anchor teacher.

The separate `offpolicy` ablation performs no rollout: it forward-noises a
dataset target latent for edit rows or a random endpoint for prompt-only T2I.

### Optional CFG-field absorption

CFG absorption is an ordinary training option, not a separate experiment
config. All released configs default to the standard conditional fields:

```yaml
training:
  teacher_cfg_scale: 1.0
  student_cfg_scale: 1.0
```

To absorb a guided teacher field into a single-pass student, keep
`student_cfg_scale=1.0` and override only `teacher_cfg_scale` (for example,
`3.5`). The teacher target becomes
`v_empty + alpha * (v_cond - v_empty)`. External inference CFG can compound
the absorbed effect, so tune the inference scale separately.

The paper defines the absorbed teacher scale and keeps the training student
field unguided. Values of `student_cfg_scale` other than `1.0` are an additional
release option, not a paper-reproduction setting.

Ordinary gradient-accumulation microbatches reuse one globally broadcast route. For the updated DiffusionOPD/FlowOPD G=M recipe, set `routing.accumulation_groups` to route dataset names; every bucket remains bound to its matching teacher and the losses are averaged in one optimizer update.

## 🚀 Quick Start

Start with the toy smoke test. It downloads the official DiffSynth example metadata, builds a tiny prompt CSV, and runs the complete DanceOPD train/update/save loop **without downloading model weights**.

```bash
pip install -e ".[smoke]"
bash scripts/bootstrap_smoke.sh
```

This is the recommended first command for a fresh checkout. It should finish in seconds and write outputs under `outputs/smoke_toy`.

### 1. Choose your first run

| Goal | Command | What it checks |
|---|---|---|
| Fastest end-to-end smoke, no weights | `bash scripts/bootstrap_smoke.sh` | data prep, router, rollout, teacher query, loss, optimizer, checkpoint save |
| Same toy smoke, direct script | `bash scripts/smoke_toy.sh` | same as above |
| SD3.5 config + data dry run | `BACKEND=sd35 bash scripts/bootstrap_smoke.sh` | dependencies, config, DiffSynth prompt extraction; no large model load |
| Z-Image config + data dry run | `BACKEND=zimage bash scripts/bootstrap_smoke.sh` | dependencies, config, DiffSynth prompt extraction; no large model load |
| Tiny SD3.5 backend training | `RUN_TRAIN=1 BACKEND=sd35 bash scripts/bootstrap_smoke.sh` | real backend load and LoRA update; first run downloads upstream weights |
| Tiny Z-Image backend training | `RUN_TRAIN=1 BACKEND=zimage bash scripts/bootstrap_smoke.sh` | real backend load and LoRA update; requires DiffSynth-Studio Z-Image install |

You can also call backend scripts directly:

```bash
bash scripts/smoke_sd35.sh --dry-run
bash scripts/smoke_zimage.sh --dry-run

# Heavier: launches the actual public model backend.
bash scripts/smoke_sd35.sh
bash scripts/smoke_zimage.sh
```

The smoke path uses the official `DiffSynth-Studio/diffsynth_example_dataset` instead of bundling custom sample data. By default it selects `z_image/Z-Image`, extracts prompts from `metadata.csv`, writes `data/diffsynth_example_dataset/danceopd_prompts.csv`, runs a 4-step rollout for 2 optimizer steps, and saves under `outputs/smoke_*`. The SD3.5 smoke uses rank 8; the Z-Image smoke uses rank 64 to match its downloadable warm-start adapter.

If a dependency is missing, the smoke scripts fail early with an install hint. To reuse an already downloaded dataset without ModelScope, set `DIFFSYNTH_NO_DOWNLOAD=1`.

Prepare only the DiffSynth prompt CSV without training:

```bash
bash scripts/prepare_diffsynth_example.sh
```

### 2. Move from smoke test to your own teachers

For real DanceOPD training, prepare a routed CSV and fill the path-free config template. This release does not include the paper teacher LoRAs or student checkpoints.

```csv
prompt
A cinematic portrait of a dancer in a softly lit studio.
A realistic product photo of a glass teapot on a wooden table.
```

An example is provided at `examples/prompts.csv`.

Fill one of the default configs:

- `configs/sd35_danceopd.yaml`
- `configs/zimage_danceopd.yaml`

Teacher fields use one shared interface:

| Teacher case | `base_ckpt` | `lora_dir` | Meaning |
|---|---|---|---|
| Base route | `null` | `null` | use the pretrained base model as a frozen teacher |
| Full checkpoint teacher | `/path/to/full_or_merged_ckpt` | `null` | load a non-LoRA teacher checkpoint |
| LoRA teacher | `null` | `/path/to/peft_lora` | merge one PEFT LoRA into the base teacher |
| Full checkpoint + LoRA | `/path/to/full_or_merged_ckpt` | `/path/to/peft_lora` | load the full checkpoint first, then merge the LoRA |

The student LoRA is created separately from frozen teachers; teacher LoRAs are
merged into clean teacher modules, not stacked on top of the student's training
adapter.

For image editing, generate the routed CSV directly from OmniEdit metadata:

```bash
python examples/prepare_omniedit.py \
  --input TIGER-Lab/OmniEdit-Filtered-1.2M \
  --output data/omniedit_danceopd.csv \
  --format danceopd \
  --max-rows 1000
```

HF input streams by default. The helper materializes source/target JPEGs and
emits `local_edit`/`global_edit` buckets; the trainer checks all buckets and
image paths before loading model weights. It recognizes the official
`src_img`, `edited_img`, `edited_prompt_list`, `task`, `o_score`, and
`omni_edit_id` fields. Style/background/environment-like tasks map to
`global_edit`; other edits map to `local_edit`. Override ambiguous categories:

```bash
python examples/prepare_omniedit.py \
  --input TIGER-Lab/OmniEdit-Filtered-1.2M \
  --output data/omniedit_danceopd.csv \
  --format danceopd \
  --task-map 'style change=global_edit,object swap=local_edit' \
  --max-rows 1000
```

Normalized DanceOPD schema:

```csv
uid,task,raw_task,prompt,source_image,target_image,caption_dict
```

Use `--format prompts` only for prompt-only T2I routes. Edit routes require a
materialized `source_image`; `offpolicy` edit training also requires
`target_image`.

To prepare normalized SFT pairs instead:

```bash
python examples/prepare_omniedit.py \
  --input TIGER-Lab/OmniEdit-Filtered-1.2M \
  --output data/omniedit_sft_pairs.csv \
  --format sft_pairs \
  --max-rows 1000
```

```csv
uid,source_image,edited_image,prompt,task,caption_dict
```

Suggested route split:

| Route | Example tasks | Purpose |
|---|---|---|
| `local_edit` | add, remove, replace, color, material, object | preservation-heavy edits |
| `global_edit` | background, environment, weather, style, tone | global transformations |
| `t2i` | prompt-only rows | base text-to-image anchor |

Key default recipe:

```yaml
training:
  method: danceopd
  resolution: 1024
  rollout_steps: 16
  k: 1
  query_bias: low_t
  lr: 2.0e-4
  grad_accum: 4
  max_train_steps: 3000
  save_steps: 300
  mixed_precision: bf16

student:
  lora_rank: 128
  lora_alpha: 128
```

### 3. Launch full training

SD3.5:

```bash
accelerate launch -m danceopd.cli.train \
  --config configs/sd35_danceopd.yaml
```

Z-Image:

```bash
accelerate launch -m danceopd.cli.train \
  --config configs/zimage_danceopd.yaml
```

Resume from any saved step (adapter + optimizer state):

```bash
accelerate launch -m danceopd.cli.train \
  --config configs/zimage_danceopd.yaml \
  --set training.resume_from=outputs/run/step-300
```

You can override paths directly from the command line:

```bash
accelerate launch -m danceopd.cli.train \
  --config configs/sd35_danceopd.yaml \
  --set model.pretrained_model='<SD35_MODEL_DIR>' \
  --set teachers.0.base_ckpt='<TEACHER_TRANSFORMER_CKPT>' \
  --set teachers.0.lora_dir='<TEACHER_LORA_DIR>' \
  --set data.prompts_csv='<PROMPTS_CSV>' \
  --set training.output_dir='<OUTPUT_DIR>'
```

Paper-template examples:

```bash
# Z-Image Local + Global Edit
accelerate launch -m danceopd.cli.train \
  --config configs/paper/zimage_edit_fusion.yaml \
  --set data.prompts_csv=data/omniedit_danceopd.csv \
  --set student.init='<STUDENT_LORA_INIT>' \
  --set teachers.0.lora_dir='<LOCAL_EDIT_LORA>' \
  --set teachers.1.lora_dir='<GLOBAL_EDIT_LORA>' \
  --set training.output_dir='<OUTPUT_DIR>'

# SD3.5-M realism-field absorption
accelerate launch -m danceopd.cli.train \
  --config configs/paper/sd35_realism_absorption.yaml \
  --set model.pretrained_model='<SD35_MODEL_DIR>' \
  --set data.prompts_csv=data/realism_prompts.csv \
  --set teachers.0.base_ckpt='<FULL_REALISM_TEACHER_CHECKPOINT>' \
  --set training.output_dir='<OUTPUT_DIR>'
```

---

## 🧩 Supported Backends

| Backend | Package path | Teacher format | Student update |
|---|---|---|---|
| Toy smoke | `danceopd.backends.toy` | deterministic prompt-derived teacher | tiny torch module |
| SD3.5 / Diffusers | `danceopd.backends.sd35_diffusers` | full transformer checkpoint and/or PEFT LoRA | PEFT LoRA |
| Z-Image / DiffSynth | `danceopd.backends.zimage_diffsynth` | DiT checkpoint and/or PEFT LoRA | PEFT LoRA |

To add a new model family, implement `DanceOPDBackend` and register it in
`danceopd/core/engine.py`:

```python
class MyBackend(DanceOPDBackend):
    def prepare(self): ...
    def compute_loss(self, sample, route): ...
    def backward(self, loss): ...
    def optimizer_step(self): ...
    def save(self, step): ...
```

The core engine owns routed data sampling, gradient accumulation, logging, and
checkpoint cadence. Use `ToyBackend` as the smallest complete example and
`SD35Backend` as the real-model reference.

---

## 📁 Repository Structure

```text
danceopd/
  core/        # algorithm: routing, rollout-state sampling, loss, trainer
  backends/    # toy smoke, SD3.5, and Z-Image adapters
  data/        # prompt CSV loader
  cli/         # training entrypoint
configs/       # path-free default configs
scripts/       # launch helpers
examples/      # DiffSynth and OmniEdit data preparation helpers
assets/        # README figures
```

---

## ✅ Dry Run

Validate config structure without loading models:

```bash
python -m danceopd.cli.train --config configs/public/sd35.yaml --dry-run
python -m danceopd.cli.train --config configs/public/zimage.yaml --dry-run
```

---

## 🔁 Reproducibility Scope

This repository releases the DanceOPD training code, public smoke tests, data adapters, and paper-style config templates. It does **not** release the paper teacher LoRAs or student checkpoints. To reproduce the full pipeline with public assets, prepare OmniEdit-style edit data, train compatible SFT teacher LoRAs or full teacher checkpoints, then run DanceOPD with those teachers.

Repository code is Apache-2.0. Upstream model and adapter weights retain their
own terms. Accept the Stable Diffusion 3.5 model license and review each Z-Image
or adapter model card before use; the repository license does not relicense
downloaded weights.

Useful docs:

- [`configs/paper/`](configs/paper/): path-free paper config templates.

---

## 📚 Citation

```bibtex
@article{zhou2026danceopd,
  title={DanceOPD: On-Policy Generative Field Distillation},
  author={Zhou, Wei and Zhu, Xiongwei and Xu, Zelin and Dong, Bo and Gong, Lixue and Liang, Yongyuan and Chu, Meng and Qu, Leigang and Kong, Lingdong and Liu, Wei and others},
  journal={arXiv preprint arXiv:2606.27377},
  year={2026}
}
```

---

## 🙏 Acknowledgements

This repository builds on PyTorch, Accelerate, Diffusers, PEFT, DiffSynth-Studio, Stable Diffusion 3.5, and Z-Image.
