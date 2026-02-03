"""
AI Translator API wrapper for fast and accurate text translation.

This module provides a simple async interface to interact with AI Translator API
through RapidAPI, allowing users to translate text between multiple languages.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    RequestError,
    InvalidInputError,
    with_retry,
)

# Configure logging
logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

# Create aliases for backward compatibility
TranslatorError = APIError
TranslatorAuthenticationError = AuthenticationError
TranslatorRequestError = RequestError
TranslatorInvalidInputError = InvalidInputError


# ==================== Data Models ====================

@dataclass
class TranslationResult(BaseResponse):
    """Result from text translation."""
    original_text: str
    translated_text: str
    word_count: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslationResult":
        """Create TranslationResult from API response dictionary."""
        return cls(
            original_text=data.get("original_text", ""),
            translated_text=data.get("translated_text", ""),
            word_count=data.get("word_count", 0)
        )

    # to_dict() and to_json() are inherited from BaseResponse


# ==================== AI Translator API Client ====================

class TranslatorAPI(BaseRapidAPI):
    """
    Async client for AI Translator API via RapidAPI.

    This API provides fast and accurate text translation using AI models.
    Supports multiple languages with high-quality translations.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with TranslatorAPI(api_key="your_key") as client:
            # Translate text
            result = await client.translate("Hello world", target_lang="ar")
            print(f"Translation: {result.translated_text}")
            print(f"Word count: {result.word_count}")

            # Translate from Arabic to English
            result = await client.translate("مرحبا بالعالم", target_lang="en")
            print(result.translated_text)

            # Use with config
            config = APIConfig(api_key="key", max_retries=5)
            async with TranslatorAPI(config=config) as client:
                result = await client.translate("text", target_lang="ar")
    """

    BASE_URL = "https://aitranslator.p.rapidapi.com"
    DEFAULT_HOST = "aitranslator.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers inherited from BaseRapidAPI

    @with_retry(max_attempts=3, delay=1.0)
    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> TranslationResult:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'en', 'ar', 'fr', 'es')
            source_lang: Source language code (optional, auto-detected if not provided)

        Returns:
            TranslationResult with original text, translation, and word count

        Raises:
            TranslatorInvalidInputError: If text is empty or invalid
            TranslatorRequestError: If request fails

        Example:
            # Translate to Arabic
            result = await client.translate("Hello world", target_lang="ar")
            print(result.translated_text)  # "مرحبا بالعالم"

            # Translate from Arabic to English
            result = await client.translate("مرحبا", target_lang="en", source_lang="ar")
            print(result.translated_text)  # "Hello"
        """
        if not text or not text.strip():
            raise InvalidInputError("Text cannot be empty")

        if not target_lang or len(target_lang) < 2:
            raise InvalidInputError("Target language code must be at least 2 characters")

        logger.info(f"Translating text to {target_lang}: {text[:50]}...")

        payload = {
            "text": text.strip(),
            "target_lang": target_lang
        }

        if source_lang:
            payload["source_lang"] = source_lang

        data = await self._make_request("POST", "/translate", json=payload)

        result = TranslationResult.from_dict(data)
        logger.info(f"Translation complete: {result.word_count} words")
        return result

    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.

        Returns:
            Dictionary with health status information

        Example:
            health = await client.health_check()
            print(f"API Status: {health['status']}")
        """
        data = await self._make_request("GET", "/health")
        logger.info("Health check successful")
        return data
