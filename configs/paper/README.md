# Paper Config Templates

These configs mirror the public DanceOPD recipe used for the paper experiments:

- hard-routed multi-teacher distillation,
- current-student rollout states,
- one low-noise query state per rollout,
- direct velocity MSE,
- rank-128 LoRA student,
- no random seed in config.

Files:

- `zimage_edit_fusion.yaml`: Z-Image / DiffSynth backend template.
- `sd35_edit_fusion.yaml`: SD3.5 / Diffusers backend template.

They intentionally contain no local paths and no teacher weights. Fill
`data.prompts_csv`, `training.output_dir`, model paths, and teacher fields before
launching. Each teacher may be a base route (`base_ckpt: null`, `lora_dir:
null`), a full/merged checkpoint (`base_ckpt` only), a PEFT LoRA (`lora_dir`
only), or a full checkpoint plus a PEFT LoRA.
