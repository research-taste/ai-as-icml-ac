from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("icml_site_build_data", ROOT / "site/build_data.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load site/build_data.py")
BUILD_DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_DATA)


MAIN = "https://openreview.net/group?id=ICML.cc/2026/Conference"
POSITION = "https://openreview.net/group?id=ICML.cc/2026/Position_Paper_Track"


def manifest_row(paper_id: str, *, tier: str = "poster", track: str = MAIN) -> dict:
    extra = {"sourceurl": track}
    if tier == "oral":
        extra["is_oral"] = True
    elif tier == "spotlight":
        extra["is_spotlight"] = True
    return {"paper_id": paper_id, "extra": extra}


def signal_row(paper_id: str, *, rank: int, contribution_class: str) -> dict:
    return {
        "paper_id": paper_id,
        "aggregate_rank": rank,
        "scores": {"contribution_profile": {"primary_contribution_class": contribution_class}},
    }


class PreferenceComparisonTests(unittest.TestCase):
    def write_signal(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / BUILD_DATA.FULL_COVERAGE_SIGNAL
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_returns_none_when_full_coverage_signal_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = BUILD_DATA.build_preference_comparison({}, scores_dir=Path(temporary))
        self.assertIsNone(result, "a clone without the analysis bundle must still build")

    def test_matches_ai_cut_to_the_size_of_the_human_honored_set(self) -> None:
        # 6 main-track papers, 2 of them honored. AI ranks theory papers top.
        manifest = {
            "a": manifest_row("a", tier="oral"),
            "b": manifest_row("b", tier="spotlight"),
            "c": manifest_row("c"),
            "d": manifest_row("d"),
            "e": manifest_row("e"),
            "f": manifest_row("f"),
            "z": manifest_row("z", track=POSITION),  # excluded: not main track
        }
        rows = [
            signal_row("a", rank=5, contribution_class="application_method"),
            signal_row("b", rank=6, contribution_class="application_method"),
            signal_row("c", rank=1, contribution_class="theory"),
            signal_row("d", rank=2, contribution_class="theory"),
            signal_row("e", rank=3, contribution_class="core_ml_algorithm"),
            signal_row("f", rank=4, contribution_class="core_ml_algorithm"),
            signal_row("z", rank=7, contribution_class="theory"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_signal(directory, rows)
            result = BUILD_DATA.build_preference_comparison(manifest, scores_dir=directory)

        assert result is not None
        self.assertEqual(result["matchedN"], 2, "AI cut must equal the honored-set size")
        sets = {entry["key"]: entry for entry in result["sets"]}
        self.assertEqual(sets["corpus"]["n"], 6, "position-track paper must be excluded")
        self.assertEqual(sets["human"]["n"], 2)
        self.assertEqual(sets["ai"]["n"], 2)
        self.assertEqual(sets["ai"]["label"], "AI's top 2")

        # Humans honored the two application papers; AI's top 2 are the theory papers.
        self.assertEqual(sets["human"]["counts"]["application_method"], 2)
        self.assertEqual(sets["ai"]["counts"]["theory"], 2)
        self.assertEqual(sets["ai"]["counts"]["application_method"], 0)

        for entry in result["sets"]:
            self.assertEqual(sum(entry["counts"].values()), entry["n"], "counts must partition the set")
            self.assertAlmostEqual(sum(entry["shares"].values()), 1.0, places=9)

    def test_refuses_to_publish_a_partial_join(self) -> None:
        manifest = {"a": manifest_row("a", tier="oral"), "b": manifest_row("b")}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_signal(directory, [signal_row("a", rank=1, contribution_class="theory")])
            with self.assertRaises(ValueError) as caught:
                BUILD_DATA.build_preference_comparison(manifest, scores_dir=directory)
        self.assertIn("misses 1 main-track", str(caught.exception))

    def test_unclassified_papers_fall_back_to_other(self) -> None:
        manifest = {"a": manifest_row("a", tier="oral"), "b": manifest_row("b")}
        rows = [
            {"paper_id": "a", "aggregate_rank": 1, "scores": {}},
            signal_row("b", rank=2, contribution_class="theory"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_signal(directory, rows)
            # 1 of 2 unclassified exceeds the 5% tolerance, so this must fail loudly.
            with self.assertRaises(ValueError) as caught:
                BUILD_DATA.build_preference_comparison(manifest, scores_dir=directory)
        self.assertIn("no contribution class", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
