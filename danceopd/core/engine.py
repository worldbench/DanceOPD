"""Generic DanceOPD training engine."""
from __future__ import annotations

import os
from pathlib import Path

from danceopd.core.config import Config, print_config_summary, require_value, validate_config
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
        validate_config(self.cfg)
        if self.dry_run:
            return
        require_value(self.cfg, "training.output_dir")
        require_value(self.cfg, "data.prompts_csv")
        backend = str(self.cfg.backend).lower()
        if backend in {"sd35", "sd3", "stable-diffusion-3.5"}:
            require_value(self.cfg, "model.pretrained_model")
        elif (
            backend in {"zimage", "z-image"}
            and not self.cfg.get_dotted("model.model_paths")
            and not self.cfg.get_dotted("model.model_id_with_origin_paths")
        ):
            raise ValueError("Z-Image requires model.model_paths or model.model_id_with_origin_paths.")

    @staticmethod
    def _require_local_resource(value, label: str) -> None:
        if value in (None, "", "base", "none", "auto", "teacher", "teacher_base"):
            return
        text = str(value)
        if text.startswith("hf://"):
            return
        # owner/repo strings are remote model IDs. Absolute paths, explicit
        # relative paths, and checkpoint filenames are local resources.
        local = os.path.isabs(text) or text.startswith(("./", "../")) or Path(text).suffix in {
            ".safetensors", ".ckpt", ".pt", ".pth", ".bin"
        }
        if local and not Path(text).expanduser().exists():
            raise ValueError(f"Missing local resource for {label}: {text}")

    def _preflight_resources(self) -> None:
        backend = str(self.cfg.backend).lower()
        if backend in {"sd35", "sd3", "stable-diffusion-3.5"}:
            self._require_local_resource(self.cfg.model.get("pretrained_model"), "model.pretrained_model")
        self._require_local_resource(self.cfg.student.get("init"), "student.init")
        resume_from = self.cfg.training.get("resume_from")
        if resume_from and not Path(str(resume_from)).expanduser().is_dir():
            raise ValueError(f"training.resume_from must be a checkpoint directory: {resume_from}")
        for index, teacher in enumerate(self.cfg.get("teachers", [])):
            self._require_local_resource(teacher.get("base_ckpt"), f"teachers.{index}.base_ckpt")
            self._require_local_resource(teacher.get("lora_dir"), f"teachers.{index}.lora_dir")
        if backend in {"zimage", "z-image"} and self.cfg.model.get("model_paths"):
            for index, path in enumerate(str(self.cfg.model.model_paths).split(",")):
                self._require_local_resource(path.strip(), f"model.model_paths[{index}]")

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
        self._preflight_resources()
        dataset.validate_routes(
            router.routes,
            require_target_for_edit=str(self.cfg.training.method).lower() == "offpolicy",
        )
        backend = build_backend(self.cfg, self.accelerator)
        backend.prepare()

        resume_from = self.cfg.training.get("resume_from")
        if resume_from:
            resume_path = Path(str(resume_from)).expanduser()
            global_step = backend.resume(str(resume_path))
        else:
            global_step = 0

        max_steps = int(self.cfg.training.max_train_steps)
        grad_accum = max(1, int(self.cfg.training.grad_accum))
        save_steps = max(1, int(self.cfg.training.save_steps))
        log_every = max(1, int(self.cfg.training.log_every))

        micro_step = 0
        while global_step < max_steps:
            # Paper-faithful hard routing: choose capability first, then draw all
            # accumulation microbatches from its matching data bucket. This keeps
            # one optimizer update semantically equal to one route (G=1).
            groups = self.cfg.routing.get("accumulation_groups")
            routes = router.accumulation_routes(groups) if self.accelerator.is_main_process else None
            if int(getattr(self.accelerator, "num_processes", 1)) > 1:
                from accelerate.utils import broadcast_object_list

                payload = [routes]
                broadcast_object_list(payload, from_process=0)
                routes = payload[0]
            losses = []
            for route in routes:
                for _ in range(grad_accum):
                    sample = dataset.sample(route.dataset)
                    if route.requires_source_image and not sample.source_image:
                        raise ValueError(
                            f"Route {route.teacher!r} requires a source image, but dataset {route.dataset!r} returned none"
                        )
                    loss = backend.compute_loss(sample=sample, route=route)
                    backend.backward(loss / (grad_accum * len(routes)))
                    losses.append(float(loss.detach().float()))
                    micro_step += 1
            backend.optimizer_step()
            global_step += 1
            if global_step % log_every == 0 and self.accelerator.is_main_process:
                print(
                    f"[DanceOPD] step={global_step}/{max_steps} "
                    f"loss={sum(losses)/len(losses):.6f} "
                    "routes=" + "+".join(f"{r.dataset}->{r.teacher}" for r in routes),
                    flush=True,
                )
            if global_step % save_steps == 0 or global_step == max_steps:
                backend.save(global_step)
        backend.close()
        if hasattr(self.accelerator, "end_training"):
            self.accelerator.end_training()
        if self.accelerator.is_main_process:
            print("[DanceOPD] done.", flush=True)
