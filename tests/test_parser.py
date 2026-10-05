import unittest

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.parser import build_compact_repr, build_scoring_repr, segment_sections, split_appendix, split_references


class ParserTests(unittest.TestCase):
    def test_split_references_uses_last_references_heading(self) -> None:
        text = "1 Introduction\nText mentions references in prose.\n\nReferences\n[1] Example"

        body, refs = split_references(text)

        self.assertIn("Introduction", body)
        self.assertNotIn("[1] Example", body)
        self.assertTrue(refs.startswith("References"))

    def test_segment_sections_finds_core_sections(self) -> None:
        text = """Abstract
This is the abstract.

1 Introduction
This is the intro.

2 Method
Details.

3 Experiments
Results.

5 Conclusion
Final remarks.
"""

        sections = segment_sections(text)

        self.assertEqual(sections["abstract"], "This is the abstract.")
        self.assertEqual(sections["introduction"], "This is the intro.")
        self.assertEqual(sections["method"], "Details.")
        self.assertEqual(sections["experiments"], "Results.")
        self.assertEqual(sections["conclusion"], "Final remarks.")

    def test_split_appendix_removes_explicit_appendix_heading(self) -> None:
        text = "1 Introduction\nMain body.\n\nAppendix A\nExtra proof."

        main, appendix = split_appendix(text)

        self.assertIn("Main body", main)
        self.assertNotIn("Extra proof", main)
        self.assertTrue(appendix.startswith("Appendix A"))

    def test_split_appendix_keeps_inline_appendix_reference(self) -> None:
        text = "Hyperparameters are discussed in\nAppendix E.\n\n4.3 Ablation Study\nAblations."

        main, appendix = split_appendix(text)

        self.assertIn("4.3 Ablation Study", main)
        self.assertEqual(appendix, "")

    def test_build_compact_repr_uses_record_metadata_and_sections(self) -> None:
        record = PaperRecord(paper_id="p1", source="accepted", title="A Paper", authors=["A", "B"])
        sections = {"abstract": "Abstract text.", "introduction": "Intro text.", "conclusion": "Conclusion text."}

        compact = build_compact_repr(record, sections, compact_char_budget=10_000)

        self.assertIn("Title: A Paper", compact)
        self.assertNotIn("Authors:", compact)
        self.assertIn("Abstract text.", compact)
        self.assertIn("Intro text.", compact)
        self.assertIn("Conclusion text.", compact)

    def test_build_scoring_repr_marks_main_body(self) -> None:
        record = PaperRecord(paper_id="p1", source="accepted", title="A Paper")

        scoring = build_scoring_repr(record, "1 Introduction\nMain paper.", scoring_char_budget=10_000)

        self.assertIn("Title: A Paper", scoring)
        self.assertIn("first 9 PDF pages", scoring)
        self.assertIn("References are removed", scoring)
        self.assertIn("Main paper", scoring)
