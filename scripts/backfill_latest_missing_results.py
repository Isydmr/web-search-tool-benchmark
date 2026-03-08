#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.utils.modifications import is_perturbed_modification

DEFAULT_DATABASE_URL = f"sqlite:///{ROOT_DIR / 'fake_news_benchmark.db'}"
DEFAULT_ARTICLE_LIMIT = 30

if TYPE_CHECKING:
    from app.models.news import News


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing model evaluations for the latest perturbed articles."
    )
    parser.add_argument("--database-url", help="Database URL for the run")
    parser.add_argument(
        "--article-limit",
        type=int,
        default=DEFAULT_ARTICLE_LIMIT,
        help="How many latest unique perturbed articles to inspect for missing model coverage",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of articles with missing coverage to backfill (0 means no cap)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of model aliases or canonical model IDs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected missing article/model pairs without calling any APIs",
    )
    return parser.parse_args()


def article_identity_parts(news: Any) -> tuple[str, str, str]:
    published_date = ""
    if isinstance(news.published_date, datetime):
        published_date = news.published_date.isoformat()
    elif news.published_date:
        published_date = str(news.published_date)
    return (
        str(news.url or news.title or "").strip(),
        str(news.source or "").strip(),
        published_date,
    )


def choose_latest_unique_articles(rows: list[Any], article_limit: int) -> list[Any]:
    selected: list[Any] = []
    seen_identities: set[tuple[str, str, str]] = set()

    for news in rows:
        if not is_perturbed_modification(news.modification_type):
            continue
        if not str(news.modified_content or "").strip():
            continue
        identity = article_identity_parts(news)
        if identity in seen_identities:
            continue
        selected.append(news)
        seen_identities.add(identity)
        if article_limit > 0 and len(selected) >= article_limit:
            break

    return selected


def resolve_models(search_api, requested_models: str | None) -> list[dict[str, str]]:
    enabled_models = search_api.get_enabled_models()
    if not requested_models:
        return enabled_models

    requested = []
    for value in requested_models.split(","):
        canonical_model_id = search_api._canonical_model_id(value.strip())
        if canonical_model_id:
            requested.append(canonical_model_id)

    requested_set = set(requested)
    return [model for model in enabled_models if model["id"] in requested_set]


def existing_model_pairs(db, News, EvaluationResult, search_api) -> dict[tuple[str, str, str], set[str]]:
    identity_by_news_id = {
        int(article.id): article_identity_parts(article)
        for article in db.query(News).all()
        if is_perturbed_modification(article.modification_type)
    }
    pairs: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for news_id, llm_model in db.query(EvaluationResult.news_id, EvaluationResult.llm_model).all():
        identity = identity_by_news_id.get(int(news_id))
        canonical_model_id = search_api._canonical_model_id(str(llm_model or ""))
        if identity and canonical_model_id:
            pairs[identity].add(canonical_model_id)

    return pairs


def choose_missing_articles(
    articles: list[Any],
    model_ids: list[str],
    existing_pairs: dict[tuple[str, str, str], set[str]],
    limit: int,
) -> list[tuple[Any, list[str]]]:
    planned: list[tuple[Any, list[str]]] = []

    for article in articles:
        identity = article_identity_parts(article)
        missing_models = [model_id for model_id in model_ids if model_id not in existing_pairs.get(identity, set())]
        if not missing_models:
            continue
        planned.append((article, missing_models))
        if limit > 0 and len(planned) >= limit:
            break

    return planned


def main() -> None:
    args = parse_args()
    os.environ["DATABASE_URL"] = args.database_url or DEFAULT_DATABASE_URL
    os.environ.setdefault("SCHEDULER_ENABLED", "0")

    from app.components.llm_evaluator import LLMEvaluator
    from app.components.llm_search_api import LLMSearchAPI
    from app.models.database import SessionLocal
    from app.models.news import EvaluationResult, News

    search_api = LLMSearchAPI()
    enabled_models = resolve_models(search_api, args.models)
    if not enabled_models:
        raise SystemExit("No enabled models are available for backfill.")

    evaluator = None if args.dry_run else LLMEvaluator()
    db = SessionLocal()
    try:
        rows = (
            db.query(News)
            .order_by(News.created_at.desc(), News.id.desc())
            .all()
        )
        latest_articles = choose_latest_unique_articles(rows, args.article_limit)
        model_ids = [model["id"] for model in enabled_models]
        existing_pairs_by_identity = existing_model_pairs(db, News, EvaluationResult, search_api)
        planned_backfills = choose_missing_articles(
            latest_articles,
            model_ids,
            existing_pairs_by_identity,
            args.limit,
        )

        evaluations_added = 0
        skipped_empty = 0
        selected_articles: list[dict[str, Any]] = []

        for article, missing_models in planned_backfills:
            selected_articles.append(
                {
                    "title": str(article.title or ""),
                    "url": str(article.url or ""),
                    "missing_models": missing_models,
                }
            )
            if args.dry_run:
                continue

            reference_text = str(article.content or "").strip()
            evaluation_input = str(article.modified_content or "").strip()
            if not reference_text or not evaluation_input:
                skipped_empty += 1
                continue

            identity = article_identity_parts(article)
            for model_id in missing_models:
                verified_content = search_api.verify_content(
                    evaluation_input,
                    model_id=model_id,
                )
                if not verified_content:
                    continue

                evaluation = evaluator._evaluate_response(reference_text, verified_content)
                db.add(
                    EvaluationResult(
                        news_id=article.id,
                        llm_model=model_id,
                        llm_response=verified_content,
                        nli_scores=evaluation.get("nli_scores", {}),
                        geval_scores=evaluation.get("geval_scores", {}),
                        semantic_similarity=evaluation.get("semantic_similarity", 0),
                        lexical_distance=evaluation.get("lexical_distance", {}),
                    )
                )
                db.commit()
                existing_pairs_by_identity[identity].add(model_id)
                evaluations_added += 1

        payload = {
            "database_url": os.environ.get("DATABASE_URL", ""),
            "article_window": args.article_limit,
            "missing_article_limit": args.limit,
            "selected_article_count": len(planned_backfills),
            "selected_articles": selected_articles,
            "models": model_ids,
            "evaluations_added": evaluations_added,
            "skipped_empty_articles": skipped_empty,
            "dry_run": args.dry_run,
        }
        print(json.dumps(payload))
    finally:
        db.close()


if __name__ == "__main__":
    main()
