import logging
import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)
from urllib3.util.retry import Retry


# ------------------------------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ------------------------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------------------------

class LLMServiceError(Exception):
    """Base exception for LLM service."""


class AuthenticationError(LLMServiceError):
    """Raised when API authentication fails."""


class RateLimitError(LLMServiceError):
    """Raised when API rate limit is exceeded."""


class LLMResponseError(LLMServiceError):
    """Raised when invalid LLM response is received."""


# ------------------------------------------------------------------------------
# LLM Service
# ------------------------------------------------------------------------------

class LLMService:
    """
    Production-grade OpenRouter LLM service.

    Features:
    - Retry strategy
    - Timeout handling
    - Connection pooling
    - Structured logging
    - Environment validation
    - Typed responses
    - Centralized session management
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize LLM service.

        Args:
            api_key: OpenRouter API key.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """

        self.api_key = api_key or os.getenv("OPEN_ROUTER_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise AuthenticationError(
                "OPEN_ROUTER_KEY environment variable is missing"
            )

        self.session = self._create_session(max_retries)

        logger.info("LLMService initialized successfully")

    # --------------------------------------------------------------------------
    # Create HTTP Session
    # --------------------------------------------------------------------------

    def _create_session(self, max_retries: int) -> requests.Session:
        """
        Create reusable requests session with retry logic.
        """

        session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],
            allowed_methods=["POST"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    # --------------------------------------------------------------------------
    # Generate Response
    # --------------------------------------------------------------------------

    def get_response(
        self,
        system_prompt: str,
        user_input: str,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate LLM response.

        Args:
            system_prompt: System instruction prompt.
            user_input: User query.
            model: OpenRouter model name.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Generated text response.

        Raises:
            LLMServiceError
        """

        try:
            # ------------------------------------------------------------------
            # Input Validation
            # ------------------------------------------------------------------

            if not system_prompt.strip():
                raise ValueError(
                    "system_prompt cannot be empty"
                )

            if not user_input.strip():
                raise ValueError(
                    "user_input cannot be empty"
                )

            logger.info(
                "Sending LLM request | model=%s",
                model,
            )

            payload: Dict = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # ------------------------------------------------------------------
            # API Call
            # ------------------------------------------------------------------

            response = self.session.post(
                url=self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # ------------------------------------------------------------------
            # Handle HTTP Errors
            # ------------------------------------------------------------------

            if response.status_code == 401:
                raise AuthenticationError(
                    "Invalid OpenRouter API key"
                )

            if response.status_code == 429:
                raise RateLimitError(
                    "OpenRouter rate limit exceeded"
                )

            response.raise_for_status()

            # ------------------------------------------------------------------
            # Parse Response
            # ------------------------------------------------------------------

            response_json = response.json()

            if "choices" not in response_json:
                raise LLMResponseError(
                    "Invalid response format received"
                )

            choices: List = response_json.get("choices", [])

            if not choices:
                raise LLMResponseError(
                    "No choices returned from LLM"
                )

            message = (
                choices[0]
                .get("message", {})
                .get("content")
            )

            if not message:
                raise LLMResponseError(
                    "Empty response content received"
                )

            logger.info(
                "LLM response generated successfully"
            )

            return message.strip()

        except (
            AuthenticationError,
            RateLimitError,
            LLMResponseError,
        ):
            logger.exception("LLM service error occurred")
            raise

        except Timeout as exc:
            logger.exception(
                "LLM request timed out"
            )

            raise LLMServiceError(
                "LLM request timeout exceeded"
            ) from exc

        except ConnectionError as exc:
            logger.exception(
                "Connection error while calling LLM API"
            )

            raise LLMServiceError(
                "Failed to connect to OpenRouter API"
            ) from exc

        except HTTPError as exc:
            logger.exception(
                "HTTP error during LLM request"
            )

            raise LLMServiceError(
                f"HTTP error occurred: {str(exc)}"
            ) from exc

        except RequestException as exc:
            logger.exception(
                "Unexpected request exception"
            )

            raise LLMServiceError(
                "Unexpected request failure"
            ) from exc

        except ValueError as exc:
            logger.exception(
                "Validation error in LLM request"
            )

            raise LLMServiceError(
                str(exc)
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during LLM generation"
            )

            raise LLMServiceError(
                "Unexpected error during LLM inference"
            ) from exc


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    try:
        llm_service = LLMService()

        response = llm_service.get_response(
            system_prompt=(
                "You are a helpful AI assistant."
            ),
            user_input=(
                "Explain vector databases in simple terms."
            ),
        )

        print("\nLLM Response:\n")
        print(response)

    except Exception:
        logger.exception(
            "Application execution failed"
        )
