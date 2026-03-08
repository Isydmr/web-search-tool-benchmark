import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from sqlalchemy.orm import Session
import pandas as pd
import threading
import uvicorn
import logging

from app.models.database import get_db, engine, Base
from app.models.news import News, EvaluationResult
from app.scheduler import Scheduler
from app.config import Config
from app.utils.benchmark_scoring import extract_geval_overall_score
from app.utils.modifications import NO_MODIFICATION_TYPE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s:%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
GENERIC_INTERNAL_ERROR = "Internal server error"

# Set all loggers to at least INFO level
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.INFO)

try:
    # Create database tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {str(e)}")
    raise

app = FastAPI(title="Fake News LLM Benchmark")

try:
    # Mount static files
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    templates = Jinja2Templates(directory="app/templates")
except Exception as e:
    logger.error(f"Error mounting static files: {str(e)}")
    raise

# Initialize scheduler for news fetching only when explicitly enabled
scheduler = None
if Config.SCHEDULER_ENABLED:
    try:
        scheduler = Scheduler()
        scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
        scheduler_thread.start()
        logger.info("Scheduler started successfully")
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
else:
    logger.info("Scheduler auto-start disabled; use scripts/run_end_to_end.sh for benchmark runs")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    try:
        news_items = (
            db.query(News)
            .filter(News.modification_type != NO_MODIFICATION_TYPE)
            .order_by(News.created_at.desc())
            .limit(Config.NEWS_FETCH_LIMIT)
            .all()
        )
        
        # Get any available evaluation results
        logger.info("Querying evaluation results...")
        results = (
            db.query(News, EvaluationResult)
            .join(EvaluationResult, News.id == EvaluationResult.news_id, isouter=True)
            .filter(News.modification_type != NO_MODIFICATION_TYPE)
            .order_by(News.created_at.desc())
            .all()
        )
        logger.info(f"Found {len(results)} total results")
        
        # Transform results into a pandas DataFrame for model scores if we have any evaluations
        data = []
        if results:
            logger.info("Processing evaluation results...")
            for news, eval_result in results:
                logger.info(f"Processing news {news.id} with eval_result: {eval_result is not None}")
                if eval_result:  # Only process if we have evaluation results
                    try:
                        data_point = {
                            "date": news.created_at.strftime("%Y-%m-%d"),
                            "model": eval_result.llm_model,
                            "source": news.source,
                            "modification": news.modification_type,
                            "nli_entailment": eval_result.nli_scores.get("entailment", 0) if eval_result.nli_scores else 0,
                            "nli_contradiction": eval_result.nli_scores.get("contradiction", 0) if eval_result.nli_scores else 0,
                            "geval_factuality": extract_geval_overall_score({"geval_scores": eval_result.geval_scores}) or 0,
                            "semantic_similarity": eval_result.semantic_similarity if eval_result.semantic_similarity else 0,
                            "lexical_distance": eval_result.lexical_distance.get("norm_edit_similarity", 0) if eval_result.lexical_distance else 0
                        }
                        data.append(data_point)
                        logger.info(f"Added data point: {data_point}")
                    except Exception as e:
                        logger.error(f"Error processing evaluation result: {str(e)}")
                        continue
        
        df = pd.DataFrame(data)
        model_scores = df.groupby("model").agg({
            "nli_entailment": "mean",
            "nli_contradiction": "mean",
            "geval_factuality": "mean",
            "semantic_similarity": "mean",
            "lexical_distance": "mean"
        }).round(3) if not df.empty else pd.DataFrame()
        model_scores_payload = (
            {
                model: {metric: float(value) for metric, value in scores.items()}
                for model, scores in model_scores.iterrows()
            }
            if not df.empty
            else {}
        )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_scores": model_scores_payload,
                "results": results if results else [],
                "news_items": news_items  # Add the news items to the template context
            }
        )
    except Exception as e:
        logger.exception("Error in index route: %s", e)
        raise HTTPException(status_code=500, detail=GENERIC_INTERNAL_ERROR)

@app.get("/api/results")
async def get_results(db: Session = Depends(get_db)):
    try:
        results = (
            db.query(News, EvaluationResult)
            .join(EvaluationResult, News.id == EvaluationResult.news_id)
            .filter(News.modification_type != NO_MODIFICATION_TYPE)
            .order_by(News.created_at.desc())
            .all()
        )
        
        return [{
            "news": {
                "id": news.id,
                "title": news.title,
                "content": news.content,
                "modified_content": news.modified_content,
                "source": news.source,
                "url": news.url,
                "modification_type": news.modification_type,
                "created_at": news.created_at
            },
            "evaluation": {
                "model": eval_result.llm_model,
                "response": eval_result.llm_response,
                "nli_scores": eval_result.nli_scores,
                "geval_scores": eval_result.geval_scores,
                "semantic_similarity": eval_result.semantic_similarity,
                "lexical_distance": eval_result.lexical_distance
            }
        } for news, eval_result in results]
    except Exception as e:
        logger.exception("Error in get_results route: %s", e)
        raise HTTPException(status_code=500, detail=GENERIC_INTERNAL_ERROR)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "scheduler_running": scheduler is not None}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=Config.APP_HOST, port=Config.APP_PORT) 
