import unittest

from scripts.publish_latest_to_viewer import (
    build_leaderboard,
    build_summary,
    filter_to_perturbed_dataset,
    limit_dataset_to_articles,
    recompute_article_stats,
)


class PublishLatestToViewerTests(unittest.TestCase):
    def test_filter_to_perturbed_dataset_drops_no_modification_rows(self):
        articles = [
            {
                "article_id": "unchanged-article",
                "title": "Original article",
                "source": "BBC",
                "modification_type": "no_modification",
                "created_at": "2026-03-08T00:00:00Z",
                "published_date": "2026-03-08T00:00:00Z",
                "evaluation_count": 1,
                "models": ["openai:gpt-5.4"],
            },
            {
                "article_id": "entity-article",
                "title": "Perturbed article",
                "source": "NPR",
                "modification_type": "entity",
                "created_at": "2026-03-08T01:00:00Z",
                "published_date": "2026-03-08T01:00:00Z",
                "evaluation_count": 1,
                "models": ["openai:gpt-5.4"],
            },
        ]
        results = [
            {
                "sample_id": "unchanged-result",
                "article_id": "unchanged-article",
                "title": "Original article",
                "source": "BBC",
                "modification_type": "no_modification",
                "model": "openai:gpt-5.4",
                "created_at": "2026-03-08T00:05:00Z",
                "nli_scores": {"entailment": 0.9, "contradiction": 0.0},
                "semantic_similarity": 0.9,
                "lexical_distance": {"norm_edit_similarity": 0.9},
                "targeted_correction": {"score": None, "status": "not_applicable"},
                "surface_alignment_score": 0.9,
                "overall_score": 0.9,
            },
            {
                "sample_id": "entity-result",
                "article_id": "entity-article",
                "title": "Perturbed article",
                "source": "NPR",
                "modification_type": "entity",
                "model": "openai:gpt-5.4",
                "created_at": "2026-03-08T01:05:00Z",
                "nli_scores": {"entailment": 0.8, "contradiction": 0.1},
                "semantic_similarity": 0.7,
                "lexical_distance": {"norm_edit_similarity": 0.6},
                "targeted_correction": {"score": 1.0, "status": "restored"},
                "surface_alignment_score": 0.75,
                "overall_score": 0.85,
            },
        ]

        filtered_articles, filtered_results = filter_to_perturbed_dataset(articles, results)
        filtered_articles = recompute_article_stats(filtered_articles, filtered_results)
        leaderboard = build_leaderboard(filtered_results)
        summary = build_summary(
            filtered_articles,
            filtered_results,
            leaderboard,
            "replace",
            "",
            [],
        )

        self.assertEqual([article["article_id"] for article in filtered_articles], ["entity-article"])
        self.assertEqual([result["sample_id"] for result in filtered_results], ["entity-result"])
        self.assertEqual(filtered_articles[0]["evaluation_count"], 1)
        self.assertEqual(summary["available_modifications"], ["entity"])
        self.assertEqual(summary["article_count"], 1)
        self.assertEqual(summary["evaluation_count"], 1)

    def test_limit_dataset_to_articles_keeps_latest_articles_and_matching_results(self):
        articles = [
            {
                "article_id": "latest-article",
                "title": "Latest perturbed article",
                "source": "BBC",
                "modification_type": "entity",
                "created_at": "2026-03-08T02:00:00Z",
                "published_date": "2026-03-08T02:00:00Z",
                "evaluation_count": 2,
                "models": ["openai:gpt-5.4", "anthropic:claude-sonnet-4-6"],
            },
            {
                "article_id": "older-article",
                "title": "Older perturbed article",
                "source": "NPR",
                "modification_type": "entity",
                "created_at": "2026-03-08T01:00:00Z",
                "published_date": "2026-03-08T01:00:00Z",
                "evaluation_count": 1,
                "models": ["perplexity:sonar"],
            },
        ]
        results = [
            {
                "sample_id": "latest-openai",
                "article_id": "latest-article",
                "title": "Latest perturbed article",
                "source": "BBC",
                "modification_type": "entity",
                "model": "openai:gpt-5.4",
                "created_at": "2026-03-08T02:05:00Z",
            },
            {
                "sample_id": "latest-anthropic",
                "article_id": "latest-article",
                "title": "Latest perturbed article",
                "source": "BBC",
                "modification_type": "entity",
                "model": "anthropic:claude-sonnet-4-6",
                "created_at": "2026-03-08T02:06:00Z",
            },
            {
                "sample_id": "older-perplexity",
                "article_id": "older-article",
                "title": "Older perturbed article",
                "source": "NPR",
                "modification_type": "entity",
                "model": "perplexity:sonar",
                "created_at": "2026-03-08T01:05:00Z",
            },
        ]

        limited_articles, limited_results = limit_dataset_to_articles(articles, results, 1)
        limited_articles = recompute_article_stats(limited_articles, limited_results)

        self.assertEqual([article["article_id"] for article in limited_articles], ["latest-article"])
        self.assertEqual(
            [result["sample_id"] for result in limited_results],
            ["latest-openai", "latest-anthropic"],
        )
        self.assertEqual(limited_articles[0]["evaluation_count"], 2)


if __name__ == "__main__":
    unittest.main()
