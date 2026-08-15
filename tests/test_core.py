import tempfile
import unittest
from pathlib import Path

import torch

from danceopd.backends.sd35_diffusers import SD35Backend
from danceopd.backends.teacher import load_compatible_state_dict
from danceopd.backends.zimage_diffsynth import ZImageBackend
from danceopd.core.config import apply_overrides, load_config, validate_config
from danceopd.core.flowopd import clipped_policy_loss, sde_step
from danceopd.core.methods import flowopd_query_indices, get_method, query_indices
from danceopd.core.routing import WeightedRouter
from danceopd.core.timestep import cfg_target, rollout_grid
from danceopd.data import PromptCSV, Sample, load_samples


class CoreTests(unittest.TestCase):
    def test_all_three_methods_are_distinct_and_validated(self):
        specs = [get_method(x) for x in ("danceopd", "diffusionopd", "flowopd")]
        self.assertEqual({x.name for x in specs}, {"danceopd", "diffusionopd", "flowopd"})
        self.assertFalse(specs[0].dense)
        self.assertTrue(specs[1].dense)
        self.assertEqual(specs[1].state_source, "student")
        self.assertEqual(specs[1].objective, "diffusion_kl")
        self.assertTrue(specs[2].stochastic_rollout)
        self.assertEqual(get_method("offpolicy").state_source, "offline")
        with self.assertRaises(ValueError):
            get_method("typo")

    def test_dense_methods_query_all_states(self):
        self.assertEqual(query_indices(get_method("diffusionopd"), 16, 1, "low_t"), list(range(16)))
        self.assertEqual(query_indices(get_method("flowopd"), 16, 1, "low_t"), list(range(16)))
        self.assertEqual(flowopd_query_indices(16, None), list(range(16)))
        self.assertEqual(len(flowopd_query_indices(16, 4)), 4)

    def test_route_is_bound_to_dataset(self):
        cfg = load_config(None)
        cfg.routing.routes = [
            {"teacher": "default", "dataset": "ocr", "weight": 1.0},
        ]
        route = WeightedRouter.from_config(cfg).sample()
        self.assertEqual(route.dataset, "ocr")
        self.assertEqual(route.teacher, "default")

    def test_csv_preserves_task_and_images(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.csv"
            p.write_text("task,prompt,source_image,target_image\nedit,make it red,src.png,dst.png\n")
            row = load_samples(str(p))[0]
            self.assertEqual(row.task, "edit")
            self.assertEqual(row.source_image, str((Path(d) / "src.png").resolve()))
            self.assertEqual(row.target_image, str((Path(d) / "dst.png").resolve()))

    def test_missing_task_bucket_fails(self):
        ds = PromptCSV([Sample("x", task="ocr")])
        with self.assertRaises(ValueError):
            ds.sample("geneval")

    def test_data_preflight_rejects_missing_image_before_model_load(self):
        ds = PromptCSV([Sample("edit it", task="edit", source_image="missing.jpg")])
        route = type("Route", (), {"dataset": "edit", "requires_source_image": True})()
        with self.assertRaisesRegex(ValueError, "Data preflight failed"):
            ds.validate_routes([route])

    def test_incompatible_full_checkpoint_is_rejected(self):
        module = torch.nn.Linear(4, 4)
        with self.assertRaisesRegex(RuntimeError, "matched 0"):
            load_compatible_state_dict(module, {"wrong": torch.ones(1)}, label="test", log=lambda _: None)

    def test_cfg_affine_field(self):
        u = torch.tensor([2.0])
        c = torch.tensor([5.0])
        self.assertEqual(cfg_target(u, c, 0).item(), 2.0)
        self.assertEqual(cfg_target(u, c, 1).item(), 5.0)
        self.assertEqual(cfg_target(u, c, 3.5).item(), 12.5)

    def test_rollout_grid_unique(self):
        self.assertEqual(rollout_grid(17, 16), list(range(17)))
        for steps in (4, 8, 16, 20, 28):
            grid = rollout_grid(steps + 1, steps)
            self.assertEqual(len(grid), steps + 1)
            self.assertEqual(len(set(grid)), len(grid))

    def test_unknown_teacher_and_method_fail_early(self):
        cfg = load_config(None)
        cfg.routing.routes = [{"teacher": "missing", "dataset": "x", "weight": 1.0}]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_flowopd_transition_and_clipped_objective_have_gradient(self):
        torch.manual_seed(0)
        z = torch.randn(2, 4, 3, 3)
        old_v = torch.zeros_like(z)
        z_next, old_log_prob, _, _ = sde_step(
            old_v, torch.tensor([0.8, 0.8]), torch.tensor([0.6, 0.6]), z, noise_level=0.7
        )
        student_v = torch.full_like(z, 0.1, requires_grad=True)
        _, log_prob, mean_student, std = sde_step(
            student_v, torch.tensor([0.8, 0.8]), torch.tensor([0.6, 0.6]), z,
            previous_sample=z_next, noise_level=0.7,
        )
        _, _, mean_teacher, _ = sde_step(
            torch.full_like(z, 0.3), torch.tensor([0.8, 0.8]), torch.tensor([0.6, 0.6]), z,
            previous_sample=z_next, noise_level=0.7,
        )
        loss, diag = clipped_policy_loss(
            log_prob=log_prob, old_log_prob=old_log_prob,
            mean_student=mean_student, mean_teacher=mean_teacher, diffusion_std=std,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(student_v.grad.abs().sum().item(), 0.0)
        self.assertGreater(diag["teacher_kl"].item(), 0.0)

    def test_g_equals_m_routes_remain_bound(self):
        cfg = load_config(None)
        cfg.training.method = "diffusionopd"
        cfg.teachers = [{"name": "ocr"}, {"name": "geneval"}]
        cfg.routing.routes = [
            {"teacher": "ocr", "dataset": "ocr", "weight": 1.0},
            {"teacher": "geneval", "dataset": "geneval", "weight": 1.0},
        ]
        cfg.routing.accumulation_groups = ["ocr", "geneval"]
        validate_config(cfg)
        routes = WeightedRouter.from_config(cfg).accumulation_routes(cfg.routing.accumulation_groups)
        self.assertEqual([(r.dataset, r.teacher) for r in routes], [("ocr", "ocr"), ("geneval", "geneval")])
        cfg = load_config(None)
        cfg.training.method = "not-a-method"
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_paper_baseline_configs_encode_reproduction_settings(self):
        root = Path(__file__).resolve().parents[1]
        baseline_dir = root / "configs/paper/baselines"
        for path in sorted(baseline_dir.glob("*.yaml")):
            with self.subTest(config=path.name):
                cfg = load_config(str(path))
                validate_config(cfg)
                if cfg.training.method == "diffusionopd":
                    self.assertEqual(cfg.training.rollout_steps, 16)
                    self.assertEqual(len(cfg.routing.accumulation_groups), 2)
                else:
                    self.assertEqual(cfg.training.method, "flowopd")
                    self.assertEqual(len(cfg.routing.accumulation_groups), 2)
                    self.assertIn("merged_init", cfg.student.init)
                    self.assertEqual(cfg.training.flowopd_group_size, 16)
                    self.assertEqual(cfg.training.flowopd_k_states, 16)
                    self.assertEqual(cfg.training.flowopd_noise_level, 0.7)
                    self.assertEqual(cfg.training.flowopd_clip_range, 1.0e-4)
                    self.assertEqual(cfg.training.flowopd_kl_scale, -1.0)
                    self.assertEqual(cfg.training.flowopd_anchor_beta, 0.0)

    def test_paper_configs_never_use_public_auto_fallback(self):
        root = Path(__file__).resolve().parents[1]
        paper_dir = root / "configs/paper"
        for path in sorted(paper_dir.rglob("*.yaml")):
            with self.subTest(config=path.relative_to(root)):
                cfg = load_config(str(path))
                self.assertNotIn(cfg.student.init, (None, "", "auto"))
                self.assertFalse(str(cfg.student.init).startswith("hf://"))
                if "unspecified_main_table" in str(cfg.student.init):
                    self.assertTrue(str(cfg.student.init).startswith("/path/to/"))

        edit_cfg = load_config(str(paper_dir / "zimage_edit_fusion.yaml"))
        self.assertTrue(all(teacher.lora_dir for teacher in edit_cfg.teachers))

    def test_package_version_matches_project_metadata(self):
        import danceopd

        root = Path(__file__).resolve().parents[1]
        metadata = (root / "pyproject.toml").read_text()
        self.assertEqual(danceopd.__version__, "0.2.0")
        self.assertIn('version = "0.2.0"', metadata)

    def test_public_defaults_do_not_make_z_student_its_teacher(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/public/zimage.yaml"))
        self.assertNotEqual(cfg.student.init, cfg.teachers[0].get("lora_dir"))
        self.assertIsNone(cfg.teachers[0].get("lora_dir"))

    def test_public_sd35_flowopd_recipe(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/public/sd35_flowopd.yaml"))
        validate_config(cfg)
        self.assertEqual(cfg.training.method, "flowopd")
        self.assertEqual(cfg.student.init, "base")
        self.assertEqual((cfg.student.lora_rank, cfg.student.lora_alpha), (32, 64))
        self.assertEqual(cfg.training.rollout_steps, 10)
        self.assertEqual(cfg.training.flowopd_group_size, 16)
        self.assertEqual(cfg.training.flowopd_k_states, 10)
        self.assertEqual([r.weight for r in WeightedRouter.from_config(cfg).routes], [1.0, 3.0])

    def test_cfg_absorption_is_a_training_option_with_unit_defaults(self):
        root = Path(__file__).resolve().parents[1]
        realism = load_config(str(root / "configs/paper/sd35_realism_absorption.yaml"))
        self.assertEqual(realism.backend, "sd35")
        self.assertEqual(realism.training.teacher_cfg_scale, 1.0)
        self.assertEqual(realism.training.student_cfg_scale, 1.0)
        self.assertFalse((root / "configs/paper/zimage_cfg_absorption.yaml").exists())

        cfg = apply_overrides(load_config(None), [
            "training.teacher_cfg_scale=3.5",
            "training.student_cfg_scale=1.0",
        ])
        validate_config(cfg)
        self.assertEqual(cfg.training.teacher_cfg_scale, 3.5)
        self.assertEqual(cfg.training.student_cfg_scale, 1.0)

    def test_cfg_scales_reject_negative_values(self):
        for key in ("teacher_cfg_scale", "student_cfg_scale"):
            cfg = load_config(None)
            cfg.training[key] = -0.1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(cfg)

        cfg = load_config(None)
        cfg.training.cfg_absorption_scale = 3.5
        with self.assertRaisesRegex(ValueError, "teacher_cfg_scale"):
            validate_config(cfg)

    def test_both_backends_construct_requested_cfg_field(self):
        cond = torch.tensor([3.0])
        uncond = torch.tensor([1.0])

        sd = object.__new__(SD35Backend)
        sd._model_velocity = lambda model, latents, timestep, embeds, pooled: embeds
        sd_out = sd._cfg_velocity(
            None, None, None, cond, None, scale=2.0,
            uncond_embeds=uncond, uncond_pooled=torch.tensor([0.0]),
        )

        z = object.__new__(ZImageBackend)
        z._velocity = lambda model, latents, timestep, embeds, image_latents=None: embeds
        z_out = z._cfg_velocity(None, None, None, cond, scale=2.0, uncond_embeds=uncond)

        self.assertEqual(sd_out.item(), 5.0)
        self.assertEqual(z_out.item(), 5.0)

    def test_zimage_edit_uses_multi_image_dit_path(self):
        cfg = load_config(None)
        cfg.training.mixed_precision = "fp32"
        cfg.training.use_gradient_checkpointing = False
        backend = object.__new__(ZImageBackend)
        backend.cfg = cfg
        backend.device = torch.device("cpu")
        backend.torch_dtype = torch.float32

        class FakeDiT:
            def __init__(self):
                self.call = None

            def __call__(self, x, timestep, prompt_embeds, **kwargs):
                self.call = (x, timestep, prompt_embeds, kwargs)
                return [x[0][-1] + x[0][0].mean()]

        model = FakeDiT()
        target = torch.ones(1, 2, 3, 3)
        source = torch.full_like(target, 2.0)
        output = backend._conditioned_velocity(
            model, target, torch.tensor(750.0), [[torch.ones(2, 4)]], [source]
        )

        self.assertEqual(model.call[3]["image_noise_mask"], [[0, 1]])
        self.assertEqual(model.call[3]["siglip_feats"], [None])
        self.assertTrue(torch.allclose(model.call[1], torch.tensor([0.25])))
        self.assertTrue(torch.allclose(output, torch.full_like(target, -3.0)))

    def test_zimage_edit_uses_omni_prompt_encoder(self):
        cfg = load_config(None)
        backend = object.__new__(ZImageBackend)
        backend.cfg = cfg
        backend.device = torch.device("cpu")
        backend.torch_dtype = torch.float32

        class PromptUnit:
            def encode_prompt(self, *args, **kwargs):
                raise AssertionError("T2I prompt encoder must not be used for edit rows")

            def encode_prompt_omni(self, pipe, prompt, edit_image, **kwargs):
                return {"prompt": prompt, "edit_image": edit_image, "kwargs": kwargs}

        pipe = type(
            "Pipe",
            (),
            {
                "units": [PromptUnit()],
                "load_models_to_device": lambda self, names: None,
            },
        )()
        backend.pipe = pipe
        marker = object()
        result = backend._encode_prompt("edit it", edit_image=marker)

        self.assertEqual(result["prompt"], "edit it")
        self.assertIs(result["edit_image"], marker)



if __name__ == "__main__":
    unittest.main()
