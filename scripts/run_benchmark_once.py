#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark once against a target database.")
    parser.add_argument("--database-url", help="Database URL for this run")
    parser.add_argument("--run-id", help="Optional run identifier")
    parser.add_argument("--run-dir", help="Optional run directory for local metadata")
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    args = parse_args()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    os.environ.setdefault("SCHEDULER_ENABLED", "0")

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    from app.config import Config

    if not Config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is required to generate modified fake-news variants.")

    from app.components.llm_search_api import LLMSearchAPI
    from app.models.database import Base, SessionLocal, engine
    from app.models.news import EvaluationResult, News
    from app.scheduler import Scheduler

    enabled_models = LLMSearchAPI().get_enabled_models()
    if not enabled_models:
        raise SystemExit(
            "At least one web-search model key is required. Configure OPENAI_API_KEY, "
            "PERPLEXITY_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY."
        )

    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        articles_before = session.query(News).count()
        results_before = session.query(EvaluationResult).count()
    finally:
        session.close()

    scheduler = Scheduler()
    scheduler.run_daily_fetch()

    session = SessionLocal()
    try:
        articles_after = session.query(News).count()
        results_after = session.query(EvaluationResult).count()
    finally:
        session.close()

    payload = {
        "run_id": args.run_id or "",
        "generated_at_utc": iso_now(),
        "database_url": Config.DATABASE_URL,
        "enabled_models": [model["id"] for model in enabled_models],
        "articles_before": articles_before,
        "articles_after": articles_after,
        "articles_added": articles_after - articles_before,
        "results_before": results_before,
        "results_after": results_after,
        "results_added": results_after - results_before,
    }

    if run_dir is not None:
        (run_dir / "run_manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
