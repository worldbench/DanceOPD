"""Generic DanceOPD training engine."""
from __future__ import annotations

from itertools import cycle

from danceopd.core.config import Config, print_config_summary, require_value, teacher_map
from danceopd.core.routing import WeightedRouter
from danceopd.data.prompt_csv import PromptCSV


def build_backend(cfg: Config, accelerator):
    name = str(cfg.backend).lower()
    if name in {"sd35", "sd3", "stable-diffusion-3.5"}:
        from danceopd.backends.sd35_diffusers import SD35Backend

        return SD35Backend(cfg, accelerator)
    if name in {"zimage", "z-image"}:
        from danceopd.backends.zimage_diffsynth import ZImageBackend

        return ZImageBackend(cfg, accelerator)
    if name in {"toy", "debug", "smoke"}:
        from danceopd.backends.toy import ToyBackend

        return ToyBackend(cfg, accelerator)
    raise ValueError(f"Unknown backend: {cfg.backend}")


class _DryRunAccelerator:
    is_main_process = True
    device = "cpu"


class DanceOPDEngine:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        if dry_run:
            self.accelerator = _DryRunAccelerator()
        else:
            from accelerate import Accelerator

            self.accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)

    def validate(self) -> None:
        teacher_map(self.cfg)
        if self.dry_run:
            return
        require_value(self.cfg, "training.output_dir")
        require_value(self.cfg, "data.prompts_csv")
        backend = str(self.cfg.backend).lower()
        if backend in {"sd35", "sd3", "stable-diffusion-3.5"}:
            require_value(self.cfg, "model.pretrained_model")
        elif backend in {"zimage", "z-image"}:
            if not self.cfg.get_dotted("model.model_paths") and not self.cfg.get_dotted("model.model_id_with_origin_paths"):
                raise ValueError("Z-Image requires model.model_paths or model.model_id_with_origin_paths.")

    def run(self) -> None:
        self.validate()
        if self.accelerator.is_main_process:
            print("[DanceOPD] " + print_config_summary(self.cfg), flush=True)
        if self.dry_run:
            if self.accelerator.is_main_process:
                print("[DanceOPD] dry run passed.", flush=True)
            return

        dataset = PromptCSV.from_config(self.cfg)
        router = WeightedRouter.from_config(self.cfg)
        backend = build_backend(self.cfg, self.accelerator)
        backend.prepare()

        max_steps = int(self.cfg.training.max_train_steps)
        grad_accum = max(1, int(self.cfg.training.grad_accum))
        save_steps = max(1, int(self.cfg.training.save_steps))
        log_every = max(1, int(self.cfg.training.log_every))

        global_step = 0
        micro_step = 0
        prompt_iter = cycle(dataset.prompts)
        while global_step < max_steps:
            prompt = next(prompt_iter)
            route = router.sample()
            loss = backend.compute_loss(prompt=prompt, route=route)
            backend.backward(loss / grad_accum)
            micro_step += 1

            if micro_step % grad_accum == 0:
                backend.optimizer_step()
                global_step += 1
                if global_step % log_every == 0 and self.accelerator.is_main_process:
                    print(
                        f"[DanceOPD] step={global_step}/{max_steps} "
                        f"loss={float(loss.detach().float()):.6f} route={route.teacher}",
                        flush=True,
                    )
                if global_step % save_steps == 0 or global_step == max_steps:
                    backend.save(global_step)
        backend.close()
        if self.accelerator.is_main_process:
            print("[DanceOPD] done.", flush=True)
