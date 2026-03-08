#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.utils.benchmark_scoring import (
    compute_overall_score,
    compute_surface_alignment_score,
    round_or_none,
    to_float,
)
from app.config import Config
from app.utils.modifications import is_perturbed_modification

POSIX_PATH_RE = re.compile(r"/Users/[^\s\"']+")
WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\\\[^\s\"']+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

TARGET_CONTEXT_WINDOW = 6
TARGET_CONTEXT_MARGIN = 0.1
MODEL_ID_ALIASES = {
    "sonar": "perplexity:sonar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a public-safe static dataset for the viewer.")
    parser.add_argument(
        "--database-url",
        default="sqlite:///./fake_news_benchmark.db",
        help="Source sqlite database URL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/latest",
        help="Published viewer dataset directory",
    )
    parser.add_argument("--run-dir", help="Optional run directory for a local snapshot copy")
    parser.add_argument("--run-id", help="Optional run identifier")
    parser.add_argument(
        "--article-limit",
        type=int,
        default=Config.PUBLISHED_ARTICLE_LIMIT,
        help=(
            "Optional limit for the number of most-recent perturbed articles to publish "
            f"(default: {Config.PUBLISHED_ARTICLE_LIMIT})"
        ),
    )
    parser.add_argument(
        "--publish-mode",
        default="auto",
        choices=["auto", "supplemental", "replace"],
        help="Merge or replace the published dataset",
    )
    return parser.parse_args()


def sanitize_string(value: str) -> str:
    return WINDOWS_PATH_RE.sub("[local-path]", POSIX_PATH_RE.sub("[local-path]", value))


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_string(value)
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_model_id(value: str) -> str:
    normalized = value.strip()
    return MODEL_ID_ALIASES.get(normalized, normalized)


def normalize_result_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_results: list[dict[str, Any]] = []
    for result in results:
        updated = dict(result)
        updated["model"] = canonical_model_id(str(updated.get("model") or "unknown"))
        normalized_results.append(updated)
    return normalized_results


def normalize_article_models(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_articles: list[dict[str, Any]] = []
    for article in articles:
        updated = dict(article)
        models = updated.get("models")
        if isinstance(models, list):
            updated["models"] = sorted(
                {canonical_model_id(str(model)) for model in models if str(model).strip()}
            )
        normalized_articles.append(updated)
    return normalized_articles


def parse_database_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise SystemExit(f"Only sqlite:/// URLs are supported right now: {database_url}")
    raw_path = database_url[len(prefix):]
    path = Path(raw_path)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def stable_id(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    for candidate in (normalized, normalized.replace(" ", "T", 1)):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def to_iso(value: Any) -> str:
    parsed = parse_datetime(value)
    if parsed is None:
        return ""
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json_field(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return sanitize_value(value)
    try:
        return sanitize_value(json.loads(value))
    except (TypeError, json.JSONDecodeError):
        return {}


def sort_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("created_at") or ""), str(record.get("title") or ""))


def load_existing_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def merge_by_key(existing: list[dict[str, Any]], incoming: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in existing:
        row_key = str(row.get(key) or "").strip()
        if row_key:
            merged[row_key] = row
    for row in incoming:
        row_key = str(row.get(key) or "").strip()
        if row_key:
            merged[row_key] = row
    return list(merged.values())


def normalize_matching_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", " ")
        .replace("–", " ")
    )
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize_for_matching(value: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_matching_text(value))


def find_subsequence_positions(tokens: list[str], needle: list[str]) -> list[int]:
    if not needle or len(needle) > len(tokens):
        return []
    return [
        index
        for index in range(len(tokens) - len(needle) + 1)
        if tokens[index : index + len(needle)] == needle
    ]


def count_entity_mentions(text: Any, entity: Any) -> int:
    return len(find_subsequence_positions(tokenize_for_matching(text), tokenize_for_matching(entity)))


def collect_entity_contexts(text: Any, entity: Any, window: int = TARGET_CONTEXT_WINDOW) -> list[tuple[list[str], list[str]]]:
    tokens = tokenize_for_matching(text)
    entity_tokens = tokenize_for_matching(entity)
    contexts: list[tuple[list[str], list[str]]] = []
    for start in find_subsequence_positions(tokens, entity_tokens):
        end = start + len(entity_tokens)
        contexts.append((tokens[max(0, start - window) : start], tokens[end : end + window]))
    return contexts


def compute_context_match_score(
    text: Any,
    entity: Any,
    contexts: list[tuple[list[str], list[str]]],
    window: int = TARGET_CONTEXT_WINDOW,
) -> float:
    tokens = tokenize_for_matching(text)
    entity_tokens = tokenize_for_matching(entity)
    if not tokens or not entity_tokens or not contexts:
        return 0.0

    best = 0.0
    for start in find_subsequence_positions(tokens, entity_tokens):
        end = start + len(entity_tokens)
        before_tokens = tokens[max(0, start - window) : start]
        after_tokens = tokens[end : end + window]
        before_set = set(before_tokens)
        after_set = set(after_tokens)
        for expected_before, expected_after in contexts:
            before_score = len(set(expected_before) & before_set) / max(1, len(set(expected_before)))
            after_score = len(set(expected_after) & after_set) / max(1, len(set(expected_after)))
            best = max(best, (before_score + after_score) / 2)
    return round(best, 4)


def compute_targeted_correction(article: dict[str, Any], response_text: str) -> dict[str, Any]:
    modification_type = str(article.get("modification_type") or "").strip().lower()
    original_entity = str(article.get("original_entity") or "").strip()
    modified_entity = str(article.get("modified_entity") or "").strip()

    if modification_type != "entity" or not original_entity or not modified_entity:
        return {
            "score": None,
            "status": "not_applicable",
            "original_entity_mentions": 0,
            "modified_entity_mentions": 0,
            "original_context_score": 0.0,
            "modified_context_score": 0.0,
        }

    contexts = collect_entity_contexts(article.get("modified_content"), modified_entity)
    original_mentions = count_entity_mentions(response_text, original_entity)
    modified_mentions = count_entity_mentions(response_text, modified_entity)
    original_context_score = compute_context_match_score(response_text, original_entity, contexts)
    modified_context_score = compute_context_match_score(response_text, modified_entity, contexts)

    if original_mentions and not modified_mentions:
        score = 1.0
        status = "restored"
    elif modified_mentions and not original_mentions:
        score = 0.0
        status = "preserved_fake"
    elif original_mentions and modified_mentions:
        if original_context_score >= modified_context_score + TARGET_CONTEXT_MARGIN:
            score = 0.85
            status = "restored_with_explanation"
        elif modified_context_score >= original_context_score + TARGET_CONTEXT_MARGIN:
            score = 0.15
            status = "preserved_fake_with_explanation"
        else:
            score = 0.5
            status = "ambiguous_mixed"
    else:
        score = 0.25
        status = "dropped_target"

    return {
        "score": round(score, 4),
        "status": status,
        "original_entity_mentions": original_mentions,
        "modified_entity_mentions": modified_mentions,
        "original_context_score": round(original_context_score, 4),
        "modified_context_score": round(modified_context_score, 4),
    }


def build_dataset(database_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not database_path.exists():
        raise SystemExit(f"Database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        article_rows = connection.execute(
            "SELECT * FROM news ORDER BY datetime(created_at) DESC, id DESC"
        ).fetchall()
        evaluation_rows = connection.execute(
            "SELECT * FROM evaluation_results ORDER BY datetime(created_at) DESC, id DESC"
        ).fetchall()
    finally:
        connection.close()

    articles_by_id: dict[str, dict[str, Any]] = {}
    article_lookup_by_news_id: dict[int, str] = {}

    for row in article_rows:
        modification_type = sanitize_string(str(row["modification_type"] or ""))
        if not is_perturbed_modification(modification_type):
            continue
        url = sanitize_string(str(row["url"] or ""))
        article_id = stable_id(
            "article",
            url or sanitize_string(str(row["title"] or "")),
            sanitize_string(str(row["source"] or "")),
            to_iso(row["published_date"]),
        )
        article_lookup_by_news_id[int(row["id"])] = article_id
        articles_by_id[article_id] = {
            "article_id": article_id,
            "title": sanitize_string(str(row["title"] or "")),
            "content": sanitize_string(str(row["content"] or "")),
            "modified_content": sanitize_string(str(row["modified_content"] or "")),
            "source": sanitize_string(str(row["source"] or "")),
            "url": url,
            "modification_type": modification_type,
            "original_entity": sanitize_string(str(row["original_entity"] or "")),
            "modified_entity": sanitize_string(str(row["modified_entity"] or "")),
            "entity_type": sanitize_string(str(row["entity_type"] or "")),
            "published_date": to_iso(row["published_date"]),
            "created_at": to_iso(row["created_at"]),
            "evaluation_count": 0,
            "models": [],
        }

    results_by_id: dict[str, dict[str, Any]] = {}
    for row in evaluation_rows:
        news_id = int(row["news_id"])
        article_id = article_lookup_by_news_id.get(news_id)
        if not article_id:
            continue
        article = articles_by_id[article_id]
        model = canonical_model_id(sanitize_string(str(row["llm_model"] or "unknown")))
        sample_id = stable_id("result", article_id, model)
        lexical_distance = load_json_field(row["lexical_distance"])
        result = {
            "sample_id": sample_id,
            "article_id": article_id,
            "title": article["title"],
            "source": article["source"],
            "url": article["url"],
            "modification_type": article["modification_type"],
            "model": model,
            "response": sanitize_string(str(row["llm_response"] or "")),
            "response_excerpt": sanitize_string(str(row["llm_response"] or ""))[:280],
            "nli_scores": load_json_field(row["nli_scores"]),
            "geval_scores": load_json_field(row["geval_scores"]),
            "semantic_similarity": round(to_float(row["semantic_similarity"]), 4),
            "lexical_distance": lexical_distance,
            "created_at": to_iso(row["created_at"]),
        }
        result["targeted_correction"] = compute_targeted_correction(article, result["response"])
        result["surface_alignment_score"] = compute_surface_alignment_score(result)
        result["overall_score"] = compute_overall_score(result)
        results_by_id[sample_id] = result

    articles = list(articles_by_id.values())
    results = list(results_by_id.values())
    return articles, results


def filter_to_perturbed_dataset(
    articles: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered_articles = [
        dict(article)
        for article in articles
        if is_perturbed_modification(article.get("modification_type"))
    ]
    allowed_article_ids = {str(article.get("article_id") or "") for article in filtered_articles}
    filtered_results = [
        dict(result)
        for result in results
        if str(result.get("article_id") or "") in allowed_article_ids
        and is_perturbed_modification(result.get("modification_type"))
    ]
    return filtered_articles, filtered_results


def limit_dataset_to_articles(
    articles: list[dict[str, Any]],
    results: list[dict[str, Any]],
    article_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if article_limit is None:
        return articles, results
    if article_limit <= 0:
        return [], []

    limited_articles = list(articles[:article_limit])
    allowed_article_ids = {str(article.get("article_id") or "") for article in limited_articles}
    limited_results = [
        dict(result)
        for result in results
        if str(result.get("article_id") or "") in allowed_article_ids
    ]
    return limited_articles, limited_results


def recompute_article_stats(
    articles: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {article["article_id"]: dict(article) for article in articles}
    models_by_article: dict[str, set[str]] = defaultdict(set)
    counts_by_article: dict[str, int] = defaultdict(int)

    for result in results:
        article_id = str(result.get("article_id") or "")
        if not article_id or article_id not in by_id:
            continue
        counts_by_article[article_id] += 1
        model = canonical_model_id(str(result.get("model") or ""))
        if model:
            models_by_article[article_id].add(model)

    updated_articles: list[dict[str, Any]] = []
    for article in by_id.values():
        article_id = article["article_id"]
        article["evaluation_count"] = counts_by_article.get(article_id, 0)
        article["models"] = sorted(models_by_article.get(article_id, set()))
        updated_articles.append(article)

    updated_articles.sort(key=sort_key, reverse=True)
    return updated_articles


def build_leaderboard(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[canonical_model_id(str(result.get("model") or "unknown"))].append(result)

    leaderboard: list[dict[str, Any]] = []
    for model, rows in grouped.items():
        nli_entailment = [to_float((row.get("nli_scores") or {}).get("entailment")) for row in rows]
        nli_contradiction = [to_float((row.get("nli_scores") or {}).get("contradiction")) for row in rows]
        semantic_values = [to_float(row.get("semantic_similarity")) for row in rows]
        lexical_values = [to_float((row.get("lexical_distance") or {}).get("norm_edit_similarity")) for row in rows]
        surface_values = [to_float(row.get("surface_alignment_score")) for row in rows]
        overall_values = [to_float(row.get("overall_score")) for row in rows]
        targeted_rows = [row for row in rows if (row.get("targeted_correction") or {}).get("score") is not None]
        targeted_values = [to_float((row.get("targeted_correction") or {}).get("score")) for row in targeted_rows]
        targeted_statuses = [str((row.get("targeted_correction") or {}).get("status") or "") for row in targeted_rows]
        targeted_count = len(targeted_rows)
        repaired_count = sum(
            1
            for status in targeted_statuses
            if status in {"restored", "restored_with_explanation"}
        )
        preserved_fake_count = sum(
            1
            for status in targeted_statuses
            if status in {"preserved_fake", "preserved_fake_with_explanation"}
        )
        dropped_target_count = targeted_statuses.count("dropped_target")
        leaderboard.append(
            {
                "model": model,
                "evaluation_count": len(rows),
                "article_count": len({row.get("article_id") for row in rows}),
                "avg_overall_score": round(sum(overall_values) / max(1, len(overall_values)), 4),
                "avg_surface_alignment_score": round(sum(surface_values) / max(1, len(surface_values)), 4),
                "targeted_sample_count": targeted_count,
                "avg_targeted_correction_score": round_or_none(
                    sum(targeted_values) / max(1, len(targeted_values)) if targeted_values else None,
                    4,
                ),
                "repair_rate": round_or_none(repaired_count / targeted_count if targeted_count else None, 4),
                "preserved_fake_rate": round_or_none(
                    preserved_fake_count / targeted_count if targeted_count else None,
                    4,
                ),
                "dropped_target_rate": round_or_none(
                    dropped_target_count / targeted_count if targeted_count else None,
                    4,
                ),
                "avg_nli_entailment": round(sum(nli_entailment) / max(1, len(nli_entailment)), 2),
                "avg_nli_contradiction": round(sum(nli_contradiction) / max(1, len(nli_contradiction)), 2),
                "avg_semantic_similarity": round(sum(semantic_values) / max(1, len(semantic_values)), 4),
                "avg_lexical_similarity": round(sum(lexical_values) / max(1, len(lexical_values)), 4),
            }
        )

    leaderboard.sort(
        key=lambda row: (
            -to_float(row.get("avg_overall_score")),
            -int(row.get("evaluation_count") or 0),
            str(row.get("model") or ""),
        )
    )
    return leaderboard


def build_summary(
    articles: list[dict[str, Any]],
    results: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    publish_mode: str,
    run_id: str,
    source_runs: list[str],
) -> dict[str, Any]:
    article_dates = [str(article.get("created_at") or "") for article in articles if article.get("created_at")]
    result_dates = [str(result.get("created_at") or "") for result in results if result.get("created_at")]
    sources = sorted({str(article.get("source") or "") for article in articles if article.get("source")})
    modifications = sorted(
        {
            str(article.get("modification_type") or "")
            for article in articles
            if article.get("modification_type")
        }
    )
    models = [canonical_model_id(str(row.get("model") or "")) for row in leaderboard if row.get("model")]
    best_repair_row = max(
        (row for row in leaderboard if row.get("avg_targeted_correction_score") is not None),
        key=lambda row: (
            to_float(row.get("avg_targeted_correction_score")),
            to_float(row.get("repair_rate")),
            -to_float(row.get("preserved_fake_rate")),
            str(row.get("model") or ""),
        ),
        default=None,
    )
    targeted_scores = [
        to_float((result.get("targeted_correction") or {}).get("score"))
        for result in results
        if (result.get("targeted_correction") or {}).get("score") is not None
    ]
    return {
        "generated_at_utc": now_iso(),
        "publish_mode": publish_mode,
        "latest_run_id": run_id,
        "source_runs": source_runs,
        "article_count": len(articles),
        "evaluation_count": len(results),
        "model_count": len(models),
        "source_count": len(sources),
        "modification_count": len(modifications),
        "available_models": models,
        "available_sources": sources,
        "available_modifications": modifications,
        "latest_article_at": max(article_dates, default=""),
        "latest_result_at": max(result_dates, default=""),
        "top_model": leaderboard[0]["model"] if leaderboard else "",
        "repairable_result_count": len(targeted_scores),
        "avg_targeted_correction_score": round_or_none(
            sum(targeted_scores) / max(1, len(targeted_scores)) if targeted_scores else None,
            4,
        ),
        "best_repair_model": best_repair_row["model"] if best_repair_row else "",
    }


def publish_dataset(
    destination: Path,
    articles: list[dict[str, Any]],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    leaderboard = build_leaderboard(results)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "articles.json", articles)
    write_json(destination / "leaderboard.json", leaderboard)
    write_json(destination / "summary.json", summary)
    write_json(destination / "manifest.json", manifest)
    write_jsonl(destination / "results.jsonl", results)


def main() -> None:
    args = parse_args()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    database_path = parse_database_path(args.database_url)

    incoming_articles, incoming_results = build_dataset(database_path)
    incoming_articles, incoming_results = filter_to_perturbed_dataset(incoming_articles, incoming_results)
    incoming_results.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    incoming_articles = recompute_article_stats(incoming_articles, incoming_results)
    incoming_articles, incoming_results = limit_dataset_to_articles(
        incoming_articles,
        incoming_results,
        args.article_limit,
    )
    incoming_articles = recompute_article_stats(incoming_articles, incoming_results)

    existing_dataset_present = (output_dir / "articles.json").exists() and (output_dir / "results.jsonl").exists()
    publish_mode = args.publish_mode
    if publish_mode == "auto":
        publish_mode = "supplemental" if existing_dataset_present else "replace"

    existing_summary = load_existing_json(output_dir / "summary.json", {}) if publish_mode == "supplemental" else {}
    existing_articles = (
        normalize_article_models(load_existing_json(output_dir / "articles.json", []))
        if publish_mode == "supplemental"
        else []
    )
    existing_results = (
        normalize_result_models(load_existing_jsonl(output_dir / "results.jsonl"))
        if publish_mode == "supplemental"
        else []
    )

    if publish_mode == "supplemental":
        merged_articles = merge_by_key(existing_articles, incoming_articles, "article_id")
        merged_results = merge_by_key(existing_results, incoming_results, "sample_id")
    else:
        merged_articles = incoming_articles
        merged_results = incoming_results

    merged_articles, merged_results = filter_to_perturbed_dataset(merged_articles, merged_results)
    merged_results = normalize_result_models(merged_results)
    merged_results.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    merged_articles = recompute_article_stats(merged_articles, merged_results)
    merged_articles, merged_results = limit_dataset_to_articles(
        merged_articles,
        merged_results,
        args.article_limit,
    )
    merged_articles = recompute_article_stats(merged_articles, merged_results)
    leaderboard = build_leaderboard(merged_results)
    incoming_leaderboard = build_leaderboard(incoming_results)

    source_runs = set(existing_summary.get("source_runs", [])) if publish_mode == "supplemental" else set()
    if args.run_id:
        source_runs.add(args.run_id)
    summary = build_summary(
        merged_articles,
        merged_results,
        leaderboard,
        publish_mode,
        args.run_id or "",
        sorted(source_runs, reverse=True),
    )
    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "publish_mode": publish_mode,
        "latest_run_id": args.run_id or "",
        "counts": {
            "articles": len(merged_articles),
            "results": len(merged_results),
            "models": len(summary["available_models"]),
        },
        "exports": {
            "summary": "summary.json",
            "leaderboard": "leaderboard.json",
            "articles": "articles.json",
            "results": "results.jsonl",
        },
    }

    publish_dataset(output_dir, merged_articles, merged_results, summary, manifest)

    snapshot_dir = None
    if run_dir is not None:
        snapshot_dir = run_dir / "viewer_snapshot"
        incoming_summary = build_summary(
            incoming_articles,
            incoming_results,
            incoming_leaderboard,
            "replace",
            args.run_id or "",
            [args.run_id] if args.run_id else [],
        )
        incoming_manifest = {
            "generated_at_utc": incoming_summary["generated_at_utc"],
            "publish_mode": "replace",
            "latest_run_id": args.run_id or "",
            "counts": {
                "articles": len(incoming_articles),
                "results": len(incoming_results),
                "models": len(incoming_summary["available_models"]),
            },
            "exports": manifest["exports"],
        }
        publish_dataset(snapshot_dir, incoming_articles, incoming_results, incoming_summary, incoming_manifest)

    payload = {
        "mode": publish_mode,
        "database_path": str(database_path),
        "output_dir": str(output_dir),
        "run_snapshot_dir": str(snapshot_dir) if snapshot_dir else "",
        "articles": len(merged_articles),
        "results": len(merged_results),
        "models": len(summary["available_models"]),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
