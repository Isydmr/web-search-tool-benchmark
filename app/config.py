import os
from dotenv import load_dotenv

load_dotenv()


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Config:
    # API Keys
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Database configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fake_news_benchmark.db")
    APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT = int(os.getenv("APP_PORT", "8080"))
    SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    
    # Default model for generating modifications
    DEFAULT_MODIFICATION_MODEL = os.getenv("DEFAULT_MODIFICATION_MODEL", "gpt-5.4")
    
    # Default OpenAI model when specifically needed
    DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-5.4")
    DEFAULT_JUDGE_MODEL = os.getenv("DEFAULT_JUDGE_MODEL", DEFAULT_OPENAI_MODEL)

    # Pinned model revisions for deterministic third-party artifact downloads
    NLI_MODEL_NAME = os.getenv(
        "NLI_MODEL_NAME",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    )
    NLI_MODEL_REVISION = os.getenv(
        "NLI_MODEL_REVISION",
        "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    )
    SENTENCE_TRANSFORMER_MODEL = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL",
        "all-mpnet-base-v2",
    )
    SENTENCE_TRANSFORMER_REVISION = os.getenv(
        "SENTENCE_TRANSFORMER_REVISION",
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    )
    
    # Scheduler settings
    SCHEDULE_TIME = "09:00"  # Daily at 9 AM
    
    # Number of news articles to fetch per benchmark run
    NEWS_FETCH_LIMIT = _int_from_env("NEWS_FETCH_LIMIT", 30)

    # Number of perturbed articles to keep in the published viewer dataset
    PUBLISHED_ARTICLE_LIMIT = _int_from_env("PUBLISHED_ARTICLE_LIMIT", 30)

    WEB_SEARCH_REQUEST_TIMEOUT_SECONDS = _int_from_env("WEB_SEARCH_REQUEST_TIMEOUT_SECONDS", 60)
    WEB_SEARCH_MAX_RETRIES = _int_from_env("WEB_SEARCH_MAX_RETRIES", 4)
    WEB_SEARCH_RETRY_BASE_DELAY_SECONDS = _float_from_env(
        "WEB_SEARCH_RETRY_BASE_DELAY_SECONDS",
        2.0,
    )
    ANTHROPIC_REQUEST_TIMEOUT_SECONDS = _int_from_env("ANTHROPIC_REQUEST_TIMEOUT_SECONDS", 120)
    ANTHROPIC_MIN_REQUEST_INTERVAL_SECONDS = _float_from_env(
        "ANTHROPIC_MIN_REQUEST_INTERVAL_SECONDS",
        3.0,
    )
    ANTHROPIC_MAX_CONTINUATIONS = _int_from_env("ANTHROPIC_MAX_CONTINUATIONS", 2)
    ANTHROPIC_TOOL_ERROR_RETRIES = _int_from_env("ANTHROPIC_TOOL_ERROR_RETRIES", 2)

    DEFAULT_WEB_SEARCH_MODELS = [
        "gpt-5.4",
        "perplexity:sonar",
        "claude-sonnet-4-6",
        "gemini-3.1-pro-preview",
    ]
    ENABLED_WEB_SEARCH_MODELS = os.getenv(
        "ENABLED_WEB_SEARCH_MODELS",
        ",".join(DEFAULT_WEB_SEARCH_MODELS),
    )
    
    # Modification types for fake news generation
    MODIFICATION_TYPES = ["entity", "date_time", "location", "number"]
    
    # TODO: Implement DAG (Directed Acyclic Graph) for task orchestration and dependency management
    # Evaluation metrics
    METRICS = ["nli", "semantic_similarity", "lexical_distance"]

    @classmethod
    def get_enabled_web_search_models(cls):
        raw_value = cls.ENABLED_WEB_SEARCH_MODELS.strip()
        if not raw_value:
            return []
        if raw_value.lower() == "all":
            return ["all"]
        return [model.strip() for model in raw_value.split(",") if model.strip()]
