import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PrepareOmniEditTests(unittest.TestCase):
    def test_official_aliases_create_routed_materialized_csv(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source.png"
            target = work / "target.png"
            Image.new("RGB", (8, 8), "red").save(source)
            Image.new("RGB", (8, 8), "blue").save(target)
            metadata = work / "rows.json"
            metadata.write_text(json.dumps([{
                "omni_edit_id": "sample-1",
                "task": "style change",
                "src_img": str(source),
                "edited_img": str(target),
                "edited_prompt_list": ["turn it into watercolor"],
                "o_score": 0.95,
            }]))
            output = work / "danceopd.csv"

            subprocess.run([
                sys.executable,
                str(root / "examples/prepare_omniedit.py"),
                "--input", str(metadata),
                "--output", str(output),
                "--format", "danceopd",
                "--max-rows", "1",
            ], check=True, cwd=root)

            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["uid"], "sample-1")
            self.assertEqual(row["task"], "global_edit")
            self.assertEqual(row["raw_task"], "style change")
            self.assertEqual(row["prompt"], "turn it into watercolor")
            self.assertTrue((output.parent / row["source_image"]).is_file())
            self.assertTrue((output.parent / row["target_image"]).is_file())

    def test_zero_usable_rows_fails(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            metadata = work / "rows.json"
            metadata.write_text('[{"task": "style"}]')
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "examples/prepare_omniedit.py"),
                    "--input",
                    str(metadata),
                    "--output",
                    str(work / "out.csv"),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No usable rows", result.stderr)


if __name__ == "__main__":
    unittest.main()
