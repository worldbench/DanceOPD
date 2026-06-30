<div align="center">

# 📘 DanceOPD Guide

### Method · Configs · Smoke Tests · Public Reproduction

<p align="center">
  <a href="https://arxiv.org/abs/2606.27377">
    <img src="https://img.shields.io/badge/arXiv-2606.27377-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper">
  </a>
  <a href="README.md">
    <img src="https://img.shields.io/badge/Main-README-111827?style=for-the-badge&logo=readme&logoColor=white" alt="README">
  </a>
  <img src="https://img.shields.io/badge/Backends-SD3.5%20%7C%20Z--Image-7C3AED?style=for-the-badge" alt="Backends">
  <img src="https://img.shields.io/badge/Weights-User%20Provided-E11D48?style=for-the-badge" alt="Weights User Provided">
</p>

This guide consolidates the public documentation for DanceOPD: the algorithm,
configuration interface, smoke tests, public data preprocessing, SFT-teacher
setup, and backend extension path.

</div>

---

## 🧭 Contents

- [Release scope](#-release-scope)
- [Method](#-method)
- [Configuration](#-configuration)
- [Smoke tests](#-smoke-tests)
- [Public data preparation](#-public-data-preparation)
- [Teacher field preparation](#-teacher-field-preparation)
- [Run DanceOPD](#-run-danceopd)
- [Custom backends](#-custom-backends)
- [Main experimental settings](#-main-experimental-settings)
- [Interpretation notes](#-interpretation-notes)

---

## 🧠 Method

DanceOPD trains one student by querying frozen teacher velocity fields on states
visited by the **current student**.

<div align="center">
<img src="assets/overview.png" width="88%" alt="DanceOPD overview">
</div>

For a route \(m\), condition \(c\), student rollout state \(z_t^\theta\), and
frozen teacher field \(v_m\):

\[
\mathcal{L}_{\text{DanceOPD}}
= \mathbb{E}_{m,c,t}\left[
\left\|v_\theta(\operatorname{sg}(z_t^\theta), t, c)
- v_m(\operatorname{sg}(z_t^\theta), t, c)\right\|_2^2
\right].
\]

<div align="center">
<img src="assets/fig2_method_query.png" width="92%" alt="DanceOPD query construction">
</div>

### Minimal update

```text
student rollout:  noise -> ... -> z_t -> ... -> clean latent
teacher query:    v_T(z_t, t, c)
student query:    v_S(z_t, t, c)
loss:             || v_S - stopgrad(v_T) ||^2
```

### Default recipe

| Component | Default |
|---|---|
| Route selection | hard route, one teacher per step |
| State source | current student trajectory |
| Query count | `K=1` |
| Query bias | low-noise / semantic-side |
| Objective | direct velocity MSE |
| Student update | LoRA |

---

## ⚙️ Configuration

The provided configs intentionally contain **no local paths**. Fill the `null`
fields before training, or override them on the command line.

```bash
danceopd-train --config configs/sd35_danceopd.yaml \
  --set model.pretrained_model='<MODEL_DIR>' \
  --set data.prompts_csv='<PROMPTS_CSV>' \
  --set training.output_dir='<OUTPUT_DIR>'
```

### Default training values

| Field | Default |
|---|---:|
| LoRA rank | 128 |
| rollout steps | 16 |
| query states `K` | 1 |
| query bias | `low_t` |
| learning rate | `2e-4` |
| grad accumulation | 4 |
| save interval | 300 |
| max train steps | 3000 |
| precision | `bf16` |

No random seed is set by the public configs.

### Paper config templates

| Template | Backend | Purpose |
|---|---|---|
| `configs/paper/zimage_edit_fusion.yaml` | Z-Image / DiffSynth | edit-fusion reproduction template |
| `configs/paper/sd35_edit_fusion.yaml` | SD3.5 / Diffusers | edit-fusion reproduction template |

Both templates are path-free. Fill `data.prompts_csv`, `training.output_dir`,
model paths, and teacher `base_ckpt` / `lora_dir` fields before launching.

---

## 🧪 Smoke Tests

### Fast first-run check

```bash
pip install -e ".[all,smoke]"
bash scripts/bootstrap_smoke.sh
```

This downloads a tiny official DiffSynth example subset and runs the full
train/update/save loop with the zero-weight toy backend. It verifies code health
without downloading large public model weights.

Expected outputs:

```text
outputs/smoke_toy/step-1/toy_student.safetensors
outputs/smoke_toy/step-2/toy_student.safetensors
```

### Real-backend dry runs

```bash
bash scripts/smoke_zimage.sh --dry-run
bash scripts/smoke_sd35.sh --dry-run
```

These validate dependency checks, config loading, prompt extraction, backend
selection, and launch plumbing without loading all model weights.

### Real-backend tiny training smoke

```bash
RUN_TRAIN=1 BACKEND=zimage bash scripts/bootstrap_smoke.sh
RUN_TRAIN=1 BACKEND=sd35 bash scripts/bootstrap_smoke.sh
```

First runs may download large upstream model weights. This is intentionally not
the default smoke path.

---

## 🗂️ Public Data Preparation

### DiffSynth official example data

```bash
bash scripts/prepare_diffsynth_example.sh
```

The helper downloads `DiffSynth-Studio/diffsynth_example_dataset`, extracts
prompts from the selected subset metadata, and writes:

```text
data/diffsynth_example_dataset/danceopd_prompts.csv
```

To reuse an already downloaded dataset:

```bash
DIFFSYNTH_NO_DOWNLOAD=1 bash scripts/prepare_diffsynth_example.sh
```

### OmniEdit-style metadata

The public helper accepts either a Hugging Face dataset ID or a local
`.csv` / `.json` / `.jsonl` metadata file.

Install dataset support:

```bash
pip install -e ".[data]"
```

Prepare edit-pair CSV for SFT teacher training:

```bash
python examples/prepare_omniedit.py \
  --input TIGER-Lab/OmniEdit-Filtered-1.2M \
  --output data/omniedit_sft_pairs.csv \
  --format sft_pairs \
  --max-rows 100000
```

Prepare prompt-only CSV for DanceOPD:

```bash
python examples/prepare_omniedit.py \
  --input TIGER-Lab/OmniEdit-Filtered-1.2M \
  --output data/omniedit_prompts.csv \
  --format prompts
```

Normalized SFT-pair schema:

```csv
uid,source_image,edited_image,prompt,task,caption_dict
```

`caption_dict` is a JSON cell containing at least `{"prompt": ...}`.

---

## 🧩 Teacher Field Preparation

DanceOPD distills **teacher velocity fields**. In the paper setting, those
teachers are SFT LoRAs trained on edit data. The public repo does not ship those
LoRAs or internal checkpoints, but it defines the interface expected by the
DanceOPD stage.

### Practical public route split

| DanceOPD route | Example OmniEdit-style tasks | Purpose |
|---|---|---|
| `local_edit` | add, remove, replace, color, material, object | local edit preservation |
| `global_edit` | background, environment, weather, style, tone | global transformations |
| `t2i` | no LoRA | base text-to-image anchor |

The exact split can differ from ours. Each route only needs to represent a
semantically meaningful frozen teacher field.

### Expected teacher config

```yaml
teachers:
  - name: local_edit
    base_ckpt: null
    lora_dir: /path/to/local_edit_lora
    weight: 1.0
  - name: global_edit
    base_ckpt: null
    lora_dir: /path/to/global_edit_lora
    weight: 1.0
  - name: t2i
    base_ckpt: null
    lora_dir: null
    weight: 1.0
```

The same interface supports LoRA and non-LoRA teachers:

| Teacher case | `base_ckpt` | `lora_dir` | Behavior |
|---|---|---|---|
| Base route | `null` | `null` | use the pretrained base model as the frozen teacher |
| Full checkpoint teacher | `/path/to/full_or_merged_ckpt` | `null` | load a full, non-LoRA teacher checkpoint |
| LoRA teacher | `null` | `/path/to/peft_lora` | load the base model, then merge one PEFT LoRA |
| Full checkpoint + LoRA | `/path/to/full_or_merged_ckpt` | `/path/to/peft_lora` | load the full checkpoint first, then merge the LoRA |

Teacher LoRAs are merged into clean frozen teacher modules. They are not stacked
on top of the student's trainable LoRA adapter.

For Z-Image, DiffSynth-Studio's Z-Image training scripts are a natural public
starting point. For SD3.5, any PEFT-compatible SD3.5 LoRA SFT trainer can be
used as long as the resulting LoRA can be loaded by PEFT.

---

## 🚀 Run DanceOPD

Example Z-Image launch:

```bash
accelerate launch -m danceopd.cli.train \
  --config configs/paper/zimage_edit_fusion.yaml \
  --set data.prompts_csv=data/omniedit_prompts.csv \
  --set teachers.0.lora_dir='<LOCAL_EDIT_LORA>' \
  --set teachers.1.lora_dir='<GLOBAL_EDIT_LORA>' \
  --set training.output_dir='<OUTPUT_DIR>'
```

Example SD3.5 launch:

```bash
accelerate launch -m danceopd.cli.train \
  --config configs/paper/sd35_edit_fusion.yaml \
  --set model.pretrained_model='<SD35_MODEL_DIR>' \
  --set data.prompts_csv=data/omniedit_prompts.csv \
  --set teachers.0.lora_dir='<LOCAL_EDIT_LORA>' \
  --set teachers.1.lora_dir='<GLOBAL_EDIT_LORA>' \
  --set training.output_dir='<OUTPUT_DIR>'
```

---

## 🛠️ Custom Backends

A backend only needs to implement `DanceOPDBackend`:

```python
class MyBackend(DanceOPDBackend):
    def prepare(self): ...
    def compute_loss(self, prompt, route): ...
    def backward(self, loss): ...
    def optimizer_step(self): ...
    def save(self, step): ...
```

The core engine handles:

- CSV prompt iteration,
- teacher route sampling,
- gradient accumulation,
- logging,
- checkpoint cadence.

Use `danceopd.backends.sd35_diffusers.SD35Backend` as the clearest real-model
reference and `danceopd.backends.toy.ToyBackend` as the smallest complete
example.

---

## 📊 Main Experimental Settings

| Component | Setting |
|---|---:|
| objective | velocity MSE |
| routing | hard route, one teacher per step |
| query source | current student rollout |
| query states | `K=1` |
| query bias | low-noise / semantic-side |
| rollout steps | `16` |
| LoRA rank | `128` |
| learning rate | `2e-4` |
| grad accumulation | `4` |
| precision | `bf16` |

---

## 💡 Interpretation Notes

- `scripts/bootstrap_smoke.sh` verifies code health, not paper quality.
- Real paper numbers require comparable teacher fields.
- Weights are intentionally user-provided.
- Teacher SFT code is not the main release path because SFT pipelines are often
  tightly coupled to base model, image-pair loader, VAE cache format, and local
  training environment.
- The DanceOPD stage is model-agnostic through the backend interface once
  compatible teacher fields are available, either as full checkpoints or PEFT LoRAs.

---

<div align="center">

**Back to [`README.md`](README.md)**

</div>
