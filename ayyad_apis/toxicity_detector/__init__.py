"""
Toxicity Detector API module for detecting toxic content in text and audio.

This module provides async access to toxicity detection for Arabic and English through RapidAPI.
"""

from .core import (
    # Main API Client
    ToxicityDetectorAPI,

    # Data Models
    BlockedWord,
    TextAnalysisResult,
    AudioAnalysisResult,

    # Exceptions
    ToxicityDetectorError,
    ToxicityAuthenticationError,
    ToxicityRequestError,
    ToxicityInvalidInputError,
)

__all__ = [
    # Main API Client
    "ToxicityDetectorAPI",

    # Data Models
    "BlockedWord",
    "TextAnalysisResult",
    "AudioAnalysisResult",

    # Exceptions
    "ToxicityDetectorError",
    "ToxicityAuthenticationError",
    "ToxicityRequestError",
    "ToxicityInvalidInputError",
]
