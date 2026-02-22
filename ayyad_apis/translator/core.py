"""
AI Translator API wrapper for text translation.

This module provides an async interface to interact with AI Translator API
through RapidAPI, allowing users to translate text between multiple languages.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

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


# ==================== Exception Aliases (Backward Compatibility) ====================

TranslatorError = APIError
TranslatorAuthenticationError = AuthenticationError
TranslatorClientError = ClientError
TranslatorRequestError = RequestError
TranslatorInvalidInputError = InvalidInputError


# ==================== Data Models ====================

@dataclass
class ChatMessage:
    """A message in a conversation chain."""
    text: str
    reply_to: Optional[str] = None


@dataclass
class TranslationResult(BaseResponse):
    """Result from text translation."""
    original_text: str
    translated_text: str
    word_count: int

    # to_dict() and to_json() inherited from BaseResponse

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslationResult":
        """Create TranslationResult from API response dictionary."""
        return cls(
            original_text=data.get("original_text", ""),
            translated_text=data.get("translated_text", ""),
            word_count=data.get("word_count", 0)
        )


@dataclass
class ConversationTranslationResult(BaseResponse):
    """Result from translation with conversation context."""
    original_text: str
    translated_text: str
    word_count: int
    context_chain: List[str] = field(default_factory=list)

    # to_dict() and to_json() inherited from BaseResponse

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTranslationResult":
        """Create ConversationTranslationResult from API response dictionary."""
        return cls(
            original_text=data.get("original_text", ""),
            translated_text=data.get("translated_text", ""),
            word_count=data.get("word_count", 0),
            context_chain=data.get("context_chain", [])
        )


# ==================== API Client ====================

class TranslatorAPI(BaseRapidAPI):
    """
    Async client for AI Translator API via RapidAPI.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with TranslatorAPI(api_key="your_key") as client:
            result = await client.translate("Hello world", target_lang="arabic")
            print(result.translated_text)

            # Translation with conversation context
            messages = {
                "1": ChatMessage(text="Did you see the game?"),
                "2": ChatMessage(text="Yeah it was amazing!", reply_to="1"),
                "3": ChatMessage(text="Who scored?", reply_to="2")
            }
            result = await client.translate_with_context(
                messages=messages,
                translate_id="3",
                target_lang="arabic"
            )
    """

    BASE_URL = "https://aitranslator.p.rapidapi.com"
    DEFAULT_HOST = "aitranslator.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers, _make_request inherited from BaseRapidAPI

    @with_retry(max_attempts=3, delay=1.0)
    async def translate(self, text: str, target_lang: str) -> TranslationResult:
        """
        Translate text to target language.

        Args:
            text: Text to translate (max 10000 characters)
            target_lang: Target language (e.g., 'arabic', 'english', 'french')

        Returns:
            TranslationResult with original_text, translated_text, word_count

        Raises:
            TranslatorInvalidInputError: If text is empty
            TranslatorRequestError: If request fails
        """
        if not text or not text.strip():
            raise InvalidInputError("Text cannot be empty")

        logger.info(f"Translating text to {target_lang}: {text[:50]}...")

        payload = {"text": text.strip(), "target_lang": target_lang}
        data = await self._make_request("POST", "/translate", json=payload)

        result = TranslationResult.from_dict(data)
        logger.info(f"Translation complete: {result.word_count} words")
        return result

    @with_retry(max_attempts=3, delay=1.0)
    async def translate_with_context(
        self,
        messages: Dict[str, ChatMessage],
        translate_id: str,
        target_lang: str
    ) -> ConversationTranslationResult:
        """
        Translate a message with conversation context.

        The API builds the conversation chain automatically by following reply_to
        references, providing better translations for pronouns and references.

        Args:
            messages: Dictionary mapping message_id -> ChatMessage
            translate_id: ID of the message to translate
            target_lang: Target language (e.g., 'arabic', 'english')

        Returns:
            ConversationTranslationResult with translation and context_chain

        Raises:
            TranslatorInvalidInputError: If translate_id not found in messages
            TranslatorRequestError: If request fails
        """
        if translate_id not in messages:
            raise InvalidInputError(f"Message ID '{translate_id}' not found")

        logger.info(f"Translating with context to {target_lang}...")

        messages_payload = {
            msg_id: {"text": msg.text, "reply_to": msg.reply_to}
            for msg_id, msg in messages.items()
        }

        payload = {
            "messages": messages_payload,
            "translate_id": translate_id,
            "target_lang": target_lang
        }

        data = await self._make_request("POST", "/translate/context", json=payload)

        result = ConversationTranslationResult.from_dict(data)
        logger.info(f"Translation with context complete: {result.word_count} words")
        return result
