import unittest
from types import SimpleNamespace

from scripts.backfill_latest_missing_results import (
    choose_latest_unique_articles,
    choose_missing_articles,
)


def make_article(
    *,
    title: str,
    url: str,
    source: str = "BBC",
    published_date: str = "2026-03-08T00:00:00Z",
    modified_content: str = "modified",
    modification_type: str = "entity",
):
    return SimpleNamespace(
        title=title,
        url=url,
        source=source,
        published_date=published_date,
        modified_content=modified_content,
        modification_type=modification_type,
    )


class BackfillLatestMissingResultsTests(unittest.TestCase):
    def test_choose_latest_unique_articles_deduplicates_by_article_identity(self):
        latest_duplicate = make_article(title="Latest", url="https://example.com/a")
        older_duplicate = make_article(title="Older duplicate", url="https://example.com/a")
        distinct_article = make_article(title="Distinct", url="https://example.com/b")

        selected = choose_latest_unique_articles(
            [latest_duplicate, older_duplicate, distinct_article],
            article_limit=10,
        )

        self.assertEqual(selected, [latest_duplicate, distinct_article])

    def test_choose_missing_articles_returns_only_models_without_existing_results(self):
        article_with_gap = make_article(title="Needs Anthropic", url="https://example.com/a")
        article_complete = make_article(title="Already Covered", url="https://example.com/b")

        existing_pairs = {
            ("https://example.com/a", "BBC", "2026-03-08T00:00:00Z"): {"openai:gpt-5.4"},
            ("https://example.com/b", "BBC", "2026-03-08T00:00:00Z"): {
                "openai:gpt-5.4",
                "anthropic:claude-sonnet-4-6",
            },
        }

        planned = choose_missing_articles(
            [article_with_gap, article_complete],
            ["openai:gpt-5.4", "anthropic:claude-sonnet-4-6"],
            existing_pairs,
            limit=10,
        )

        self.assertEqual(
            planned,
            [(article_with_gap, ["anthropic:claude-sonnet-4-6"])],
        )


if __name__ == "__main__":
    unittest.main()
