import schedule
import time
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.config import Config
from app.components.news_fetcher import NewsFetcher
from app.components.llm_modifier import LLMModifier
from app.components.llm_search_api import LLMSearchAPI
from app.components.llm_evaluator import LLMEvaluator
from app.models.database import get_db
from app.models.news import News, EvaluationResult
from app.utils.modifications import is_perturbed_modification

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.news_modifier = LLMModifier()
        self.search_api = LLMSearchAPI()
        self.evaluator = LLMEvaluator()
        logger.info("Initialized all LLM components")
        
    def run_daily_fetch(self):
        db = None
        try:
            logger.info("Starting daily news fetch")
            # Get database session
            db = next(get_db())
            enabled_models = self.search_api.get_enabled_models()
            logger.info(
                "Running benchmark with %d web search models: %s",
                len(enabled_models),
                ", ".join(model["id"] for model in enabled_models) if enabled_models else "none",
            )
            
            # Fetch news
            news_articles = self.news_fetcher.fetch_top_stories(limit=Config.NEWS_FETCH_LIMIT)
            logger.info(f"Fetched {len(news_articles)} news articles")

            existing_news_by_url = {
                news.url: news
                for news in db.query(News).all()
                if is_perturbed_modification(news.modification_type)
            }
            existing_eval_pairs = {
                (news_id, llm_model)
                for news_id, llm_model in db.query(EvaluationResult.news_id, EvaluationResult.llm_model).all()
            }
            logger.info(
                "Loaded %d existing articles and %d existing article/model evaluations",
                len(existing_news_by_url),
                len(existing_eval_pairs),
            )

            created_articles = 0
            created_evaluations = 0
            for article in news_articles:
                try:
                    news = existing_news_by_url.get(article["url"])
                    if news is None:
                        modified_content, modification_type, original_entity, modified_entity, entity_type = (
                            self.news_modifier.modify_news(article["content"])
                        )
                        if not is_perturbed_modification(modification_type):
                            logger.info("Skipping article without a perturbation: %s", article["title"])
                            continue
                        news = News(
                            title=article["title"],
                            content=article["content"],
                            source=article["source"],
                            url=article["url"],
                            published_date=article.get("published_date", datetime.utcnow()),
                            modified_content=modified_content,
                            modification_type=modification_type,
                            original_entity=original_entity,
                            modified_entity=modified_entity,
                            entity_type=entity_type,
                        )
                        db.add(news)
                        db.commit()
                        db.refresh(news)
                        existing_news_by_url[news.url] = news
                        created_articles += 1
                        logger.info("Added article: %s", article["title"])
                    else:
                        logger.info("Reusing existing article: %s", news.title)

                    evaluation_input = news.modified_content or article["content"]
                    reference_text = news.content or article["content"]

                    for model_config in enabled_models:
                        model_id = model_config["id"]
                        try:
                            if (news.id, model_id) in existing_eval_pairs:
                                logger.info(
                                    "Skipping existing %s evaluation for article: %s",
                                    model_id,
                                    news.title,
                                )
                                continue

                            logger.info(
                                "Starting %s verification for article: %s",
                                model_id,
                                news.title,
                            )
                            verified_content = self.search_api.verify_content(
                                evaluation_input,
                                model_id=model_id,
                            )
                            if not verified_content:
                                logger.warning(
                                    "Model %s returned no verification output for article: %s",
                                    model_id,
                                    news.title,
                                )
                                continue

                            evaluation = self.evaluator._evaluate_response(
                                reference_text,
                                verified_content,
                            )
                            eval_result = EvaluationResult(
                                news_id=news.id,
                                llm_model=model_id,
                                llm_response=verified_content,
                                nli_scores=evaluation.get('nli_scores', {}),
                                geval_scores=evaluation.get('geval_scores', {}),
                                semantic_similarity=evaluation.get('semantic_similarity', 0),
                                lexical_distance=evaluation.get('lexical_distance', {})
                            )
                            db.add(eval_result)
                            db.commit()
                            existing_eval_pairs.add((news.id, model_id))
                            created_evaluations += 1
                            logger.info(
                                "Added %s evaluation for article: %s",
                                model_id,
                                news.title,
                            )
                        except Exception as model_error:
                            logger.error(
                                "Error evaluating article '%s' with model %s: %s",
                                news.title,
                                model_id,
                                model_error,
                            )
                            db.rollback()
                except Exception as e:
                    logger.error(f"Error adding news article: {str(e)}")
                    db.rollback()

            logger.info(
                "Daily fetch finished with %d new articles and %d new evaluations",
                created_articles,
                created_evaluations,
            )
                    
        except Exception as e:
            logger.error(f"Error in run_daily_fetch: {str(e)}")
        finally:
            if db is not None:
                db.close()
    
    def start(self):
        logger.info("Starting scheduler")
        # Add initial delay to allow services to properly initialize
        startup_delay = 30  # seconds
        logger.info(f"Waiting {startup_delay} seconds for services to initialize...")
        time.sleep(startup_delay)
        max_retries = 3
        retry_delay = 10  # seconds

        # Try to run initial fetch with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Initial fetch attempt {attempt + 1}/{max_retries}")
                self.run_daily_fetch()
                break
            except Exception as e:
                logger.error(f"Error in initial fetch attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("All initial fetch attempts failed")
        
        # Schedule daily run
        schedule.every().day.at(Config.SCHEDULE_TIME).do(self.run_daily_fetch)
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)
