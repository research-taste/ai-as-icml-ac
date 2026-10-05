from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("icml_site_build", ROOT / "site/build.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load site/build.py")
SITE_BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SITE_BUILD)


class SiteBuildTests(unittest.TestCase):
    def test_fresh_clone_build_uses_committed_data_and_creates_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "icml-ai-ac-blog"
            report = SITE_BUILD.build_blog(output=output)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["papers"], 60)
            self.assertTrue(report["used_committed_data"])
            self.assertEqual(
                (output / "site/data.js").read_bytes(),
                (ROOT / "site/data.js").read_bytes(),
            )
            self.assertTrue((output / "docs/stage7_human_comparison.md").is_file())
            archive = Path(report["archive"])
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as handle:
                names = set(handle.namelist())
            self.assertIn("icml-ai-ac-blog/site/index.html", names)
            self.assertIn("icml-ai-ac-blog/site/data.js", names)
            self.assertIn("icml-ai-ac-blog/docs/final_pipeline_methodology.md", names)

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "blog"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                SITE_BUILD.build_blog(output=output, create_archive=False)


if __name__ == "__main__":
    unittest.main()
