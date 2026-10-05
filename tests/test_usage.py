import unittest

from icml_ai_ac.scoring.usage import sum_usage


class UsageTests(unittest.TestCase):
    def test_sum_usage_preserves_float_cost_and_ignores_nested_fields(self) -> None:
        result = sum_usage(
            [
                {"prompt_tokens": 10, "cost": 0.25, "is_byok": False, "details": {"x": 1}},
                {"prompt_tokens": 20, "cost": 0.5, "is_byok": True},
            ]
        )

        self.assertEqual(result, {"prompt_tokens": 30, "cost": 0.75})


if __name__ == "__main__":
    unittest.main()
