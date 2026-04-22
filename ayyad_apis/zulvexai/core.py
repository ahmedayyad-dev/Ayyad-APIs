"""
ZulvexAI API wrapper for AI chat.

This module provides an async interface to interact with ZulvexAI API
through RapidAPI, allowing users to send prompts and receive AI responses.

Author: Ahmed Ayyad
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

logger = logging.getLogger(__name__)


# ==================== Exception Aliases ====================

ZulvexAIError = APIError
ZulvexAIAuthenticationError = AuthenticationError
ZulvexAIClientError = ClientError
ZulvexAIRequestError = RequestError
ZulvexAIInvalidInputError = InvalidInputError


# ==================== Data Models ====================

@dataclass
class ChatResult(BaseResponse):
    """Result from ZulvexAI chat request."""
    prompt: str
    response: str

    # to_dict() and to_json() inherited from BaseResponse

    @classmethod
    def from_dict(cls, prompt: str, data: Dict[str, Any]) -> "ChatResult":
        """Create ChatResult from API response dictionary."""
        # The API may return the response under different keys
        response_text = (
            data.get("response")
            or data.get("message")
            or data.get("text")
            or data.get("content")
            or str(data)
        )
        return cls(
            prompt=prompt,
            response=response_text,
        )


# ==================== API Client ====================

class ZulvexAIAPI(BaseRapidAPI):
    """
    Async client for ZulvexAI API via RapidAPI.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with ZulvexAIAPI(api_key="your_key") as client:
            result = await client.chat("What is the capital of Egypt?")
            print(result.response)
    """

    BASE_URL = "https://zulvexai.p.rapidapi.com"
    DEFAULT_HOST = "zulvexai.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers, _make_request inherited from BaseRapidAPI

    @with_retry(max_attempts=3, delay=1.0)
    async def chat(self, prompt: str) -> ChatResult:
        """
        Send a prompt to ZulvexAI and receive an AI response.

        Args:
            prompt: The text prompt to send to the AI

        Returns:
            ChatResult with the prompt and AI response

        Raises:
            ZulvexAIInvalidInputError: If prompt is empty
            ZulvexAIRequestError: If request fails
        """
        if not prompt or not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")

        logger.info(f"Sending prompt to ZulvexAI: {prompt}...")

        payload = {"prompt": prompt.strip()}
        data = await self._make_request("POST", "/chat", json=payload)

        result = ChatResult.from_dict(prompt, data)
        logger.info("ZulvexAI response received")
        return result
