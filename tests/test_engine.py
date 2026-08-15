import tempfile
import unittest
from pathlib import Path

from danceopd.core.config import apply_overrides, load_config
from danceopd.core.engine import DanceOPDEngine


class EngineTests(unittest.TestCase):
    def test_toy_end_to_end_all_methods(self):
        root = Path(__file__).resolve().parents[1]
        cfg_path = root / "configs/smoke/toy_diffsynth_example.yaml"
        for method in ("danceopd", "diffusionopd", "flowopd", "offpolicy"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as out:
                prompt_path = Path(out) / "prompts.txt"
                prompt_path.write_text("prompt\na small test image\n")
                cfg = apply_overrides(load_config(str(cfg_path)), [
                    f"data.prompts_csv={prompt_path}",
                    f"training.output_dir={out}",
                    f"training.method={method}",
                    "training.max_train_steps=1",
                    "training.save_steps=1",
                ])
                DanceOPDEngine(cfg).run()
                self.assertTrue((Path(out) / "step-1/toy_student.safetensors").exists())

    def test_diffusionopd_g_equals_m_update(self):
        root = Path(__file__).resolve().parents[1]
        cfg_path = root / "configs/smoke/toy_diffsynth_example.yaml"
        with tempfile.TemporaryDirectory() as out:
            prompt_path = Path(out) / "prompts.csv"
            prompt_path.write_text("task,prompt\nocr,write CAT\ngeneval,two cubes\n")
            cfg = apply_overrides(load_config(str(cfg_path)), [
                f"data.prompts_csv={prompt_path}",
                f"training.output_dir={out}",
                "training.method=diffusionopd",
                "training.max_train_steps=1",
                "training.save_steps=1",
            ])
            cfg.teachers = [{"name": "ocr"}, {"name": "geneval"}]
            cfg.routing.routes = [
                {"teacher": "ocr", "dataset": "ocr"},
                {"teacher": "geneval", "dataset": "geneval"},
            ]
            cfg.routing.accumulation_groups = ["ocr", "geneval"]
            DanceOPDEngine(cfg).run()
            self.assertTrue((Path(out) / "step-1/toy_student.safetensors").exists())

    def test_toy_checkpoint_resumes_optimizer_and_step(self):
        root = Path(__file__).resolve().parents[1]
        cfg_path = root / "configs/smoke/toy_diffsynth_example.yaml"
        with tempfile.TemporaryDirectory() as out:
            prompt_path = Path(out) / "prompts.csv"
            prompt_path.write_text("prompt\nresume test\n")
            base_overrides = [
                f"data.prompts_csv={prompt_path}",
                f"training.output_dir={out}",
                "training.save_steps=1",
                "training.max_train_steps=1",
            ]
            DanceOPDEngine(apply_overrides(load_config(str(cfg_path)), base_overrides)).run()
            first = Path(out) / "step-1"
            self.assertTrue((first / "trainer_state.pt").exists())

            resumed = base_overrides[:-1] + [
                "training.max_train_steps=2",
                f"training.resume_from={first}",
            ]
            DanceOPDEngine(apply_overrides(load_config(str(cfg_path)), resumed)).run()
            self.assertTrue((Path(out) / "step-2/toy_student.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
