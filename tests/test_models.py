import unittest

from icml_ai_ac.models import PaperRecord


class PaperRecordTests(unittest.TestCase):
    def test_from_dict_normalizes_nullable_container_fields(self) -> None:
        record = PaperRecord.from_dict(
            {
                "paper_id": "p1",
                "source": "accepted",
                "authors": None,
                "extra": None,
                "unknown": "ignored",
            }
        )

        self.assertEqual(record.authors, [])
        self.assertEqual(record.extra, {})
        self.assertFalse(hasattr(record, "unknown"))

    def test_from_dict_wraps_single_author_string(self) -> None:
        record = PaperRecord.from_dict({"paper_id": "p1", "source": "accepted", "authors": "Ada"})

        self.assertEqual(record.authors, ["Ada"])


if __name__ == "__main__":
    unittest.main()
