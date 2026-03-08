import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import requests

from app.config import Config

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai.types import GenerateContentConfig, GoogleSearch, Tool  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    genai = None
    GenerateContentConfig = None
    GoogleSearch = None
    Tool = None


logger = logging.getLogger(__name__)

FACT_CHECK_SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Use live web search results to compare the given "
    "text with reliable sources and provide a corrected version that fixes inaccuracies "
    "while preserving the original structure. Return only the corrected article text. "
    "Do not add prefaces, explanations, bullet lists, annotations, or markdown fences."
)


class LLMSearchAPI:
    MODEL_REGISTRY = {
        "openai:gpt-5.4": {
            "provider": "openai",
            "model": "gpt-5.4",
            "method": "_verify_with_openai_responses",
            "api_key_attr": "openai_api_key",
        },
        "openai:gpt-5": {
            "provider": "openai",
            "model": "gpt-5",
            "method": "_verify_with_openai_responses",
            "api_key_attr": "openai_api_key",
        },
        "openai:gpt-4o-search-preview": {
            "provider": "openai",
            "model": "gpt-4o-search-preview",
            "method": "_verify_with_openai_chat_search",
            "api_key_attr": "openai_api_key",
        },
        "openai:gpt-4o-mini-search-preview": {
            "provider": "openai",
            "model": "gpt-4o-mini-search-preview",
            "method": "_verify_with_openai_chat_search",
            "api_key_attr": "openai_api_key",
        },
        "perplexity:sonar": {
            "provider": "perplexity",
            "model": "sonar",
            "method": "_verify_with_perplexity",
            "api_key_attr": "perplexity_api_key",
        },
        "perplexity:sonar-pro": {
            "provider": "perplexity",
            "model": "sonar-pro",
            "method": "_verify_with_perplexity",
            "api_key_attr": "perplexity_api_key",
        },
        "perplexity:sonar-reasoning-pro": {
            "provider": "perplexity",
            "model": "sonar-reasoning-pro",
            "method": "_verify_with_perplexity",
            "api_key_attr": "perplexity_api_key",
        },
        "perplexity:sonar-deep-research": {
            "provider": "perplexity",
            "model": "sonar-deep-research",
            "method": "_verify_with_perplexity",
            "api_key_attr": "perplexity_api_key",
        },
        "anthropic:claude-opus-4-6": {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "method": "_verify_with_anthropic",
            "api_key_attr": "anthropic_api_key",
        },
        "anthropic:claude-opus-4-5-20251101": {
            "provider": "anthropic",
            "model": "claude-opus-4-5-20251101",
            "method": "_verify_with_anthropic",
            "api_key_attr": "anthropic_api_key",
        },
        "anthropic:claude-opus-4-1-20250805": {
            "provider": "anthropic",
            "model": "claude-opus-4-1-20250805",
            "method": "_verify_with_anthropic",
            "api_key_attr": "anthropic_api_key",
        },
        "anthropic:claude-sonnet-4-6": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "method": "_verify_with_anthropic",
            "api_key_attr": "anthropic_api_key",
        },
        "anthropic:claude-sonnet-4-5-20250929": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "method": "_verify_with_anthropic",
            "api_key_attr": "anthropic_api_key",
        },
        "anthropic:claude-sonnet-4-20250514": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "method": "_verify_with_anthropic",
            "api_key_attr": "anthropic_api_key",
        },
        "anthropic:claude-haiku-4-5-20251001": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "method": "_verify_with_anthropic",
            "api_key_attr": "anthropic_api_key",
        },
        "google:gemini-2.5-flash": {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "method": "_verify_with_google",
            "api_key_attr": "google_api_key",
        },
        "google:gemini-2.5-pro": {
            "provider": "google",
            "model": "gemini-2.5-pro",
            "method": "_verify_with_google",
            "api_key_attr": "google_api_key",
        },
        "google:gemini-3.1-pro-preview": {
            "provider": "google",
            "model": "gemini-3.1-pro-preview",
            "method": "_verify_with_google",
            "api_key_attr": "google_api_key",
        },
    }

    MODEL_ALIASES = {
        "gpt-5.4": "openai:gpt-5.4",
        "gpt-5": "openai:gpt-5",
        "gpt-4o-search-preview": "openai:gpt-4o-search-preview",
        "gpt-4o-mini-search-preview": "openai:gpt-4o-mini-search-preview",
        "sonar": "perplexity:sonar",
        "sonar-pro": "perplexity:sonar-pro",
        "sonar-reasoning-pro": "perplexity:sonar-reasoning-pro",
        "sonar-deep-research": "perplexity:sonar-deep-research",
        "claude-opus-4-6": "anthropic:claude-opus-4-6",
        "claude-opus-4-5-20251101": "anthropic:claude-opus-4-5-20251101",
        "claude-opus-4-1-20250805": "anthropic:claude-opus-4-1-20250805",
        "claude-sonnet-4-6": "anthropic:claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929": "anthropic:claude-sonnet-4-5-20250929",
        "claude-sonnet-4-20250514": "anthropic:claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001": "anthropic:claude-haiku-4-5-20251001",
        "gemini-2.5-flash": "google:gemini-2.5-flash",
        "gemini-2.5-pro": "google:gemini-2.5-pro",
        "gemini-3.1": "google:gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview": "google:gemini-3.1-pro-preview",
        "g3.1": "google:gemini-3.1-pro-preview",
    }

    def __init__(self):
        self.openai_api_key = Config.OPENAI_API_KEY
        self.perplexity_api_key = Config.PERPLEXITY_API_KEY
        self.anthropic_api_key = Config.ANTHROPIC_API_KEY
        self.google_api_key = Config.GOOGLE_API_KEY
        self.google_client = self._build_google_client()
        self._last_request_started_at: dict[str, float] = {}

    def _build_google_client(self):
        if not self.google_api_key:
            return None
        if genai is None:
            logger.warning("google-genai is not installed; Gemini web search models will be skipped")
            return None
        return genai.Client(api_key=self.google_api_key)

    @classmethod
    def _canonical_model_id(cls, model_id: str) -> Optional[str]:
        if not model_id:
            return None
        normalized = model_id.strip()
        if normalized in cls.MODEL_REGISTRY:
            return normalized
        return cls.MODEL_ALIASES.get(normalized)

    def get_enabled_models(self) -> List[Dict[str, str]]:
        requested_models = Config.get_enabled_web_search_models()
        if requested_models == ["all"]:
            requested_models = list(self.MODEL_REGISTRY.keys())

        enabled_models = []
        seen_models = set()

        for requested_model in requested_models:
            canonical_model_id = self._canonical_model_id(requested_model)
            if not canonical_model_id:
                logger.warning("Unknown web search model configured: %s", requested_model)
                continue
            if canonical_model_id in seen_models:
                continue

            model_config = self.MODEL_REGISTRY[canonical_model_id]
            if not self._model_is_available(model_config):
                logger.info(
                    "Skipping %s because the required API key or dependency is missing",
                    canonical_model_id,
                )
                continue

            enabled_models.append({"id": canonical_model_id, **model_config})
            seen_models.add(canonical_model_id)

        return enabled_models

    def _model_is_available(self, model_config: Dict[str, str]) -> bool:
        api_key = getattr(self, model_config["api_key_attr"], None)
        if not api_key:
            return False
        if model_config["provider"] == "google":
            return self.google_client is not None
        return True

    def _post_json_with_retries(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict,
        provider_name: str,
        timeout: Optional[int] = None,
        min_interval_seconds: float = 0.0,
    ) -> Optional[Dict]:
        max_retries = max(1, Config.WEB_SEARCH_MAX_RETRIES)
        timeout_seconds = timeout or Config.WEB_SEARCH_REQUEST_TIMEOUT_SECONDS

        for attempt in range(max_retries):
            self._respect_min_interval(provider_name, min_interval_seconds)
            try:
                self._last_request_started_at[provider_name] = time.monotonic()
                response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
                if response.status_code == 200:
                    return response.json()

                if response.status_code in {408, 409, 429, 500, 502, 503, 504, 529} and attempt < max_retries - 1:
                    delay_seconds = self._retry_delay_seconds(response, attempt)
                    logger.warning(
                        "%s API returned %s; retrying in %.1fs",
                        provider_name,
                        response.status_code,
                        delay_seconds,
                    )
                    time.sleep(delay_seconds)
                    continue

                request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
                logger.error("%s API error: %s", provider_name, response.status_code)
                if request_id:
                    logger.error("%s request id: %s", provider_name, request_id)
                if response.text:
                    logger.error("%s API response: %s", provider_name, response.text)
                return None
            except requests.RequestException as exc:
                logger.error("%s request failed: %s", provider_name, exc)
                if attempt < max_retries - 1:
                    delay_seconds = self._retry_delay_seconds(None, attempt)
                    logger.warning("%s request retry in %.1fs", provider_name, delay_seconds)
                    time.sleep(delay_seconds)
                    continue
                return None

        return None

    def _respect_min_interval(self, provider_name: str, min_interval_seconds: float) -> None:
        if min_interval_seconds <= 0:
            return
        last_started_at = self._last_request_started_at.get(provider_name)
        if last_started_at is None:
            return
        elapsed = time.monotonic() - last_started_at
        remaining = min_interval_seconds - elapsed
        if remaining > 0:
            logger.info("Waiting %.2fs before the next %s request", remaining, provider_name)
            time.sleep(remaining)

    def _retry_delay_seconds(self, response: Optional[requests.Response], attempt: int) -> float:
        base_delay = max(0.0, Config.WEB_SEARCH_RETRY_BASE_DELAY_SECONDS)
        exponential_delay = base_delay * (2 ** attempt)
        header_delay = 0.0

        if response is not None:
            header_delay = self._retry_after_seconds(response.headers.get("retry-after"))

        return max(base_delay, exponential_delay, header_delay)

    @staticmethod
    def _retry_after_seconds(raw_value: Optional[str]) -> float:
        if not raw_value:
            return 0.0

        try:
            return max(0.0, float(raw_value))
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(raw_value)
        except (TypeError, ValueError, IndexError):
            return 0.0

        if retry_at.tzinfo is None:
            return 0.0

        return max(0.0, retry_at.timestamp() - time.time())

    @staticmethod
    def _extract_anthropic_text(response_json: Dict[str, Any]) -> str:
        response_text_parts = []
        for content_block in response_json.get("content", []):
            if content_block.get("type") == "text" and content_block.get("text"):
                response_text_parts.append(content_block["text"])
        return "\n".join(response_text_parts).strip()

    @staticmethod
    def _anthropic_tool_error_type(response_json: Dict[str, Any]) -> Optional[str]:
        for content_block in response_json.get("content", []):
            if content_block.get("type") == "web_search_tool_result_error":
                return str(content_block.get("error_type") or "web_search_tool_result_error")
        return None

    def _verify_with_openai_responses(self, modified_content: str, model_name: str) -> str:
        if not self.openai_api_key:
            return ""

        payload = {
            "model": model_name,
            "tools": [{"type": "web_search"}],
            "instructions": FACT_CHECK_SYSTEM_PROMPT,
            "input": modified_content,
            "max_output_tokens": 1500,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        response_json = self._post_json_with_retries(
            "https://api.openai.com/v1/responses",
            headers,
            payload,
            provider_name="OpenAI",
        )
        if not response_json:
            return ""

        if response_json.get("output_text"):
            return response_json["output_text"].strip()

        response_text_parts = []
        for output_item in response_json.get("output", []):
            if output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []):
                if content_item.get("type") in {"output_text", "text"} and content_item.get("text"):
                    response_text_parts.append(content_item["text"])

        return "\n".join(response_text_parts).strip()

    def _verify_with_openai_chat_search(self, modified_content: str, model_name: str) -> str:
        if not self.openai_api_key:
            return ""

        payload = {
            "model": model_name,
            "web_search_options": {},
            "messages": [
                {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": modified_content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        response_json = self._post_json_with_retries(
            "https://api.openai.com/v1/chat/completions",
            headers,
            payload,
            provider_name="OpenAI",
        )
        if not response_json:
            return ""

        choices = response_json.get("choices", [])
        if not choices:
            return ""

        return choices[0].get("message", {}).get("content", "").strip()

    def _verify_with_perplexity(self, modified_content: str, model_name: str) -> str:
        if not self.perplexity_api_key:
            return ""

        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": modified_content},
            ],
            "max_tokens": 1500,
        }
        response_json = self._post_json_with_retries(
            "https://api.perplexity.ai/chat/completions",
            headers,
            payload,
            provider_name="Perplexity",
        )
        if not response_json:
            return ""

        choices = response_json.get("choices", [])
        if not choices:
            return ""

        return choices[0].get("message", {}).get("content", "").strip()

    def _verify_with_anthropic(self, modified_content: str, model_name: str) -> str:
        if not self.anthropic_api_key:
            return ""

        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        base_payload = {
            "model": model_name,
            "max_tokens": 1500,
            "system": FACT_CHECK_SYSTEM_PROMPT,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                }
            ],
        }
        messages: list[Dict[str, Any]] = [{"role": "user", "content": modified_content}]
        tool_error_attempts = 0
        continuations = 0

        while True:
            payload = {**base_payload, "messages": messages}
            response_json = self._post_json_with_retries(
                "https://api.anthropic.com/v1/messages",
                headers,
                payload,
                provider_name="Anthropic",
                timeout=Config.ANTHROPIC_REQUEST_TIMEOUT_SECONDS,
                min_interval_seconds=Config.ANTHROPIC_MIN_REQUEST_INTERVAL_SECONDS,
            )
            if not response_json:
                return ""

            tool_error_type = self._anthropic_tool_error_type(response_json)
            if tool_error_type:
                if tool_error_attempts >= max(0, Config.ANTHROPIC_TOOL_ERROR_RETRIES):
                    logger.error("Anthropic web search tool kept failing with %s", tool_error_type)
                    return ""
                tool_error_attempts += 1
                delay_seconds = max(
                    Config.ANTHROPIC_MIN_REQUEST_INTERVAL_SECONDS,
                    self._retry_delay_seconds(None, tool_error_attempts - 1),
                )
                logger.warning(
                    "Anthropic web search tool error (%s); retrying request in %.1fs",
                    tool_error_type,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
                continue

            response_text = self._extract_anthropic_text(response_json)
            if response_text:
                return response_text

            if response_json.get("stop_reason") == "pause_turn":
                if continuations >= max(0, Config.ANTHROPIC_MAX_CONTINUATIONS):
                    logger.error("Anthropic response paused too many times without producing text")
                    return ""
                assistant_content = response_json.get("content", [])
                if not assistant_content:
                    logger.error("Anthropic paused without assistant content to continue")
                    return ""
                messages.append({"role": "assistant", "content": assistant_content})
                continuations += 1
                logger.info("Continuing Anthropic pause_turn response (%d/%d)", continuations, Config.ANTHROPIC_MAX_CONTINUATIONS)
                continue

            return ""

    def _verify_with_google(self, modified_content: str, model_name: str) -> str:
        if not self.google_client or not GenerateContentConfig or not GoogleSearch or not Tool:
            return ""

        try:
            prompt = f"{FACT_CHECK_SYSTEM_PROMPT}\n\nArticle to fact-check:\n{modified_content}"
            response = self.google_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[Tool(google_search=GoogleSearch())],
                ),
            )
            return (getattr(response, "text", "") or "").strip()
        except Exception as exc:  # pragma: no cover
            logger.error("Google Gemini API error: %s", exc)
            return ""

    def verify_content(
        self,
        modified_content: str,
        api: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        logger.info("Verifying content using model=%s api=%s", model_id, api)

        if model_id is None:
            if api == "openai":
                model_id = "gpt-5.4"
            elif api == "perplexity":
                model_id = "perplexity:sonar"
            elif api == "anthropic":
                model_id = "claude-sonnet-4-6"
            elif api == "google":
                model_id = "gemini-3.1-pro-preview"
            else:
                model_id = "perplexity:sonar"

        canonical_model_id = self._canonical_model_id(model_id)
        if not canonical_model_id:
            raise ValueError(f"Unsupported model: {model_id}")

        model_config = self.MODEL_REGISTRY[canonical_model_id]
        if not self._model_is_available(model_config):
            logger.warning("Skipping unavailable web search model: %s", canonical_model_id)
            return ""

        verifier = getattr(self, model_config["method"])
        return verifier(modified_content, model_config["model"])
