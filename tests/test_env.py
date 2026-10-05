import os
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.env import load_dotenv, strip_env_value


class EnvTests(unittest.TestCase):
    def test_strip_env_value_removes_matching_quotes(self) -> None:
        self.assertEqual(strip_env_value('"abc"'), "abc")
        self.assertEqual(strip_env_value("'abc'"), "abc")
        self.assertEqual(strip_env_value("abc"), "abc")

    def test_load_dotenv_does_not_override_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("TEST_ENV_KEY=from_file\n", encoding="utf-8")
            os.environ["TEST_ENV_KEY"] = "existing"
            try:
                loaded = load_dotenv(path)
                self.assertEqual(os.environ["TEST_ENV_KEY"], "existing")
                self.assertEqual(loaded, {})
            finally:
                os.environ.pop("TEST_ENV_KEY", None)
