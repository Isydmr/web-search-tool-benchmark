import unittest

from app.utils.benchmark_scoring import (
    compute_overall_score,
    compute_surface_alignment_score,
    extract_geval_overall_score,
    normalize_geval_overall,
)


class BenchmarkScoringTests(unittest.TestCase):
    def test_normalize_geval_overall_handles_likert_scale(self):
        self.assertEqual(normalize_geval_overall(1), 0.0)
        self.assertEqual(normalize_geval_overall(5), 1.0)
        self.assertEqual(normalize_geval_overall(0.75), 0.75)

    def test_extract_geval_overall_prefers_normalized_field(self):
        self.assertEqual(
            extract_geval_overall_score({"geval_scores": {"normalized_overall": 0.6, "overall": 5}}),
            0.6,
        )

    def test_surface_alignment_is_judge_led_when_available(self):
        score = compute_surface_alignment_score(
            {
                "geval_scores": {"normalized_overall": 1.0},
                "semantic_similarity": 0.2,
                "lexical_distance": {"norm_edit_similarity": 0.3},
                "nli_scores": {"entailment": 80.0, "contradiction": 10.0},
            }
        )
        self.assertAlmostEqual(score, 0.78, places=4)

    def test_overall_score_blends_surface_and_targeted_repair(self):
        score = compute_overall_score(
            {
                "surface_alignment_score": 0.78,
                "targeted_correction": {"score": 0.25},
            }
        )
        self.assertAlmostEqual(score, 0.568, places=4)


if __name__ == "__main__":
    unittest.main()
