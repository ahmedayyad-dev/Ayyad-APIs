"""
AI Translator API module for fast and accurate text translation.

This module provides async access to AI-powered translation through RapidAPI.
"""

from .core import (
    # Main API Client
    TranslatorAPI,

    # Data Models
    TranslationResult,

    # Exceptions
    TranslatorError,
    TranslatorAuthenticationError,
    TranslatorClientError,
    TranslatorRequestError,
    TranslatorInvalidInputError,
)

__all__ = [
    # Main API Client
    "TranslatorAPI",

    # Data Models
    "TranslationResult",

    # Exceptions
    "TranslatorError",
    "TranslatorAuthenticationError",
    "TranslatorClientError",
    "TranslatorRequestError",
    "TranslatorInvalidInputError",
]
