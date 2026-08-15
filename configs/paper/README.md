# Paper topology configs

These configs encode paper-aligned topology and reported algorithm settings.
Unreleased teacher/student checkpoints are explicit placeholders and are not
silently replaced. A placeholder marked `unspecified_main_table` means that the
manuscript does not state that initialization for the corresponding main-table
run; users must supply the checkpoint used by their reproduction.

- `zimage_t2i_edit.yaml`: main T2I + joint-Edit two-bucket experiment.
- `zimage_edit_fusion.yaml`: main Local + Global two-bucket experiment.
- `zimage_three_bucket_diagnostic.yaml`: three-bucket diagnostic only.
- `sd35_realism_absorption.yaml`: SD3.5-M realism-field absorption with the
  unreleased full-parameter realism teacher left as an explicit placeholder.
- `baselines/zimage_*_diffusionopd.yaml`: both Table-2 DiffusionOPD blocks with
  dense `K=16` and explicit same-step `G=M=2` accumulation.
- `baselines/zimage_*_flowopd.yaml`: both Table-2 Flow-OPD blocks with merged
  initialization, group size 16, dense `K=16`, `eta=0.7`, KL scale `-1`, PPO
  clip `1e-4`, and MAR disabled.

For a smoke run using downloadable checkpoints, use `configs/public/{sd35,zimage}.yaml` instead.

CFG absorption is a training option, not a separate topology config. Every
config defaults to `training.teacher_cfg_scale=1.0` and
`training.student_cfg_scale=1.0`. To absorb a guided teacher field, keep the
student at `1.0` and override only the teacher scale, for example
`training.teacher_cfg_scale=3.5`.

Do not reproduce Table 2 by changing only `training.method`: use the explicit
baseline template so the reported algorithm and teacher-scheduling settings are
preserved. Initialization placeholders remain explicit where the manuscript is
underspecified.
Set `training.flowopd_mar_teacher` with a positive
`training.flowopd_anchor_beta` only when a compatible MAR anchor is available.
