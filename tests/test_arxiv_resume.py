import unittest

from icml_ai_ac.cli import _arxiv_resolution_run_ids, _arxiv_row_needs_retry


class ArxivResumeTests(unittest.TestCase):
    def test_retry_campaign_skips_row_already_attempted_in_same_campaign(self) -> None:
        row = {
            "extra": {
                "arxiv_resolution_status": "unresolved",
                "arxiv_resolution_run_id": "refresh-1",
            }
        }

        self.assertFalse(
            _arxiv_row_needs_retry(
                row,
                retry_failed=False,
                retry_unresolved=True,
                resolution_run_id="refresh-1",
            )
        )
        self.assertTrue(
            _arxiv_row_needs_retry(
                row,
                retry_failed=False,
                retry_unresolved=True,
                resolution_run_id="refresh-2",
            )
        )

    def test_retry_without_campaign_preserves_legacy_behavior(self) -> None:
        row = {"extra": {"arxiv_resolution_status": "failed"}}

        self.assertTrue(
            _arxiv_row_needs_retry(
                row,
                retry_failed=True,
                retry_unresolved=False,
                resolution_run_id=None,
            )
        )

    def test_retry_campaign_checks_attempt_history(self) -> None:
        row = {
            "extra": {
                "arxiv_resolution_status": "unresolved",
                "arxiv_resolution_run_id": "targeted-retry",
                "arxiv_resolution_run_ids": ["refresh-1", "targeted-retry"],
            }
        }

        self.assertEqual(_arxiv_resolution_run_ids(row), {"refresh-1", "targeted-retry"})
        self.assertFalse(
            _arxiv_row_needs_retry(
                row,
                retry_failed=False,
                retry_unresolved=True,
                resolution_run_id="refresh-1",
            )
        )


if __name__ == "__main__":
    unittest.main()
