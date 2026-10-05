import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.storage import read_jsonl


class StorageTests(unittest.TestCase):
    def test_read_jsonl_requires_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"ok": true}\n["not", "an", "object"]\n', encoding="utf-8")

            reader = read_jsonl(path)
            self.assertEqual(next(reader), {"ok": True})
            with self.assertRaisesRegex(ValueError, "JSONL row must be an object"):
                next(reader)

    def test_read_jsonl_reports_invalid_json_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"ok": true}\n{bad json}\n', encoding="utf-8")

            reader = read_jsonl(path)
            self.assertEqual(next(reader), {"ok": True})
            with self.assertRaisesRegex(ValueError, r":2: invalid JSON"):
                next(reader)


if __name__ == "__main__":
    unittest.main()
