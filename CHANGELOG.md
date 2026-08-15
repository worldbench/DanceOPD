# Changelog

## [0.2.0] - 2026-08-15
### Features
- Added task-bound teacher routing and structured edit samples.
- Added DanceOPD, DiffusionOPD, and FlowOPD modes for SD3.5-M and Z-Image.
- Added runnable public configs, paper-aligned templates, checkpoint resume, dual-LoRA support, and CFG-field training options.
- Consolidated method, configuration, data-preparation, teacher, and backend-extension documentation into `README.md`.

### Design Rationale
- Route selection happens before data sampling so each task is supervised by its matching teacher.
- Downloadable fallback assets are separated from paper templates to avoid presenting substitute weights as exact reproduction assets.
- A single README avoids duplicated documentation drifting out of sync.

### Notes & Caveats
- Paper metrics require compatible teacher fields and evaluation assets.
- SD3.5-M requires access to the upstream gated checkpoint.
- Z-Image requires the documented DiffSynth-Studio interface.
