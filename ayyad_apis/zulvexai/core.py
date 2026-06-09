"""
ZulvexAI API wrapper for AI chat.

Provides async access to multiple AI providers via RapidAPI.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    ClientError,
    RequestError,
    InvalidInputError,
    with_retry,
)
from ..types import DeepSeekModelType

logger = logging.getLogger(__name__)


# ==================== Exception Aliases ====================

ZulvexAIError = APIError
ZulvexAIAuthenticationError = AuthenticationError
ZulvexAIClientError = ClientError
ZulvexAIRequestError = RequestError
ZulvexAIInvalidInputError = InvalidInputError


# ==================== Data Model ====================


@dataclass
class ChatResult(BaseResponse):
    """Response from ZulvexAI chat provider."""
    reply: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatResult":
        return cls(reply=data.get("reply", ""))


# ==================== API Client ====================


class ZulvexAIAPI(BaseRapidAPI):
    """
    Async client for ZulvexAI API via RapidAPI.

    Supports multiple AI providers:
    - deepseek
    - gemini
    - chatgpt
    - mistral

    Example:
        async with ZulvexAIAPI(api_key="your_key") as client:
            result = await client.deepseek(
                "Explain quantum computing",
                thinking_enabled=True,
            )
            print(result.reply)
    """

    BASE_URL = "https://zulvexai.p.rapidapi.com"
    DEFAULT_HOST = "zulvexai.p.rapidapi.com"

    def _validate_prompt(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")
        return prompt.strip()

    async def _chat(self, endpoint: str, prompt: str, **extra: Any) -> ChatResult:
        prompt = self._validate_prompt(prompt)
        payload: Dict[str, Any] = {"prompt": prompt}
        payload.update(extra)
        data = await self._make_request("POST", endpoint, json=payload)
        return ChatResult.from_dict(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def deepseek(
        self,
        prompt: str,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        model_type: DeepSeekModelType = "default",
    ) -> ChatResult:
        """Send a prompt to DeepSeek AI."""
        return await self._chat(
            "/deepseek", prompt,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            model_type=model_type,
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def gemini(self, prompt: str) -> ChatResult:
        """Send a prompt to Google Gemini."""
        return await self._chat("/gemini", prompt)

    @with_retry(max_attempts=3, delay=1.0)
    async def chatgpt(self, prompt: str) -> ChatResult:
        """Send a prompt to ChatGPT."""
        return await self._chat("/chatgpt", prompt)

    @with_retry(max_attempts=3, delay=1.0)
    async def mistral(
        self,
        prompt: str,
        search_enabled: bool = False,
    ) -> ChatResult:
        """Send a prompt to Mistral AI."""
        return await self._chat(
            "/mistral", prompt,
            search_enabled=search_enabled,
        )
