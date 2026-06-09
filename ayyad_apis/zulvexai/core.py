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


# ==================== Data Models ====================


@dataclass
class DeepSeekResult(BaseResponse):
    """Response from DeepSeek chat."""
    reply: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepSeekResult":
        return cls(reply=data.get("reply", ""))


@dataclass
class GeminiResult(BaseResponse):
    """Response from Gemini chat."""
    reply: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiResult":
        return cls(reply=data.get("reply", ""))


@dataclass
class ChatGPTResult(BaseResponse):
    """Response from ChatGPT."""
    reply: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatGPTResult":
        return cls(reply=data.get("reply", ""))


@dataclass
class MistralResult(BaseResponse):
    """Response from Mistral chat."""
    reply: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MistralResult":
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

    @with_retry(max_attempts=3, delay=1.0)
    async def deepseek(
        self,
        prompt: str,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        model_type: DeepSeekModelType = "default",
    ) -> DeepSeekResult:
        """
        Send a prompt to DeepSeek AI.

        Args:
            prompt: The message to send to the AI
            thinking_enabled: Enable deep thinking mode
            search_enabled: Enable web search
            model_type: Model variant - "default", "expert", or "vision"

        Returns:
            DeepSeekResult with the AI response
        """
        prompt = self._validate_prompt(prompt)
        payload = {
            "prompt": prompt,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "model_type": model_type,
        }
        data = await self._make_request("POST", "/deepseek", json=payload)
        return DeepSeekResult.from_dict(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def gemini(self, prompt: str) -> GeminiResult:
        """
        Send a prompt to Google Gemini.

        Args:
            prompt: The message to send to the AI

        Returns:
            GeminiResult with the AI response
        """
        prompt = self._validate_prompt(prompt)
        data = await self._make_request("POST", "/gemini", json={"prompt": prompt})
        return GeminiResult.from_dict(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def chatgpt(self, prompt: str) -> ChatGPTResult:
        """
        Send a prompt to ChatGPT.

        Args:
            prompt: The message to send to the AI

        Returns:
            ChatGPTResult with the AI response
        """
        prompt = self._validate_prompt(prompt)
        data = await self._make_request("POST", "/chatgpt", json={"prompt": prompt})
        return ChatGPTResult.from_dict(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def mistral(
        self,
        prompt: str,
        search_enabled: bool = False,
    ) -> MistralResult:
        """
        Send a prompt to Mistral AI.

        Args:
            prompt: The message to send to the AI
            search_enabled: Enable web search

        Returns:
            MistralResult with the AI response
        """
        prompt = self._validate_prompt(prompt)
        payload = {
            "prompt": prompt,
            "search_enabled": search_enabled,
        }
        data = await self._make_request("POST", "/mistral", json=payload)
        return MistralResult.from_dict(data)
