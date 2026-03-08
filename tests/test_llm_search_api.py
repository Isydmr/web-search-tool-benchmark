import unittest
from unittest.mock import patch

from app.components.llm_search_api import LLMSearchAPI


class LLMSearchAPITests(unittest.TestCase):
    def test_retry_after_seconds_supports_numeric_header(self):
        self.assertEqual(LLMSearchAPI._retry_after_seconds("7"), 7.0)

    @patch("app.components.llm_search_api.time.time", return_value=100.0)
    def test_retry_after_seconds_supports_http_date(self, mocked_time):
        delay = LLMSearchAPI._retry_after_seconds("Thu, 01 Jan 1970 00:01:45 GMT")
        self.assertEqual(delay, 5.0)
        mocked_time.assert_called_once()

    def test_anthropic_tool_error_type_detects_web_search_error_blocks(self):
        response_json = {
            "content": [
                {"type": "server_tool_use", "name": "web_search"},
                {"type": "web_search_tool_result_error", "error_type": "max_uses_exceeded"},
            ]
        }

        self.assertEqual(
            LLMSearchAPI._anthropic_tool_error_type(response_json),
            "max_uses_exceeded",
        )


if __name__ == "__main__":
    unittest.main()
