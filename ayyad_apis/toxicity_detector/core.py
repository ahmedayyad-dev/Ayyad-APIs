"""
Toxicity Detector API wrapper for detecting toxic content in text and audio.

This module provides a simple async interface to interact with Toxicity Detector API
through RapidAPI, allowing users to detect toxic language in Arabic and English.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from pathlib import Path

import aiohttp
import aiofiles

# Import base classes and utilities
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

# Configure logging
logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

# Create aliases for backward compatibility
ToxicityDetectorError = APIError
ToxicityAuthenticationError = AuthenticationError
ToxicityClientError = ClientError
ToxicityRequestError = RequestError
ToxicityInvalidInputError = InvalidInputError


# ==================== Data Models ====================

@dataclass
class ObfuscatedWord(BaseResponse):
    """Represents an obfuscated word with original and corrected forms."""
    original: str
    corrected: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObfuscatedWord":
        """Create ObfuscatedWord from API response dictionary."""
        return cls(
            original=data.get("original", ""),
            corrected=data.get("corrected", "")
        )

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class TextAnalysisResult(BaseResponse):
    """Result from text toxicity analysis."""
    blocked: bool
    confidence: float
    obfuscated_words: List[ObfuscatedWord]
    message: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextAnalysisResult":
        """Create TextAnalysisResult from API response dictionary."""
        obfuscated_words: List[ObfuscatedWord] = []
        if "obfuscated_words" in data and isinstance(data["obfuscated_words"], list):
            obfuscated_words = [ObfuscatedWord.from_dict(w) for w in data["obfuscated_words"]]

        return cls(
            blocked=data.get("blocked", False),
            confidence=data.get("confidence", 0.0),
            obfuscated_words=obfuscated_words,
            message=data.get("message", "")
        )

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class AudioAnalysisResult(BaseResponse):
    """Result from audio toxicity analysis."""
    text: str
    blocked: bool
    confidence: float
    obfuscated_words: List[ObfuscatedWord]
    message: str = ""

    def __post_init__(self) -> None:
        if self.obfuscated_words is None:
            self.obfuscated_words = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioAnalysisResult":
        """Create AudioAnalysisResult from API response dictionary."""
        obfuscated_words: List[ObfuscatedWord] = []
        if "obfuscated_words" in data and isinstance(data["obfuscated_words"], list):
            obfuscated_words = [ObfuscatedWord.from_dict(w) for w in data["obfuscated_words"]]

        return cls(
            text=data.get("text", ""),
            blocked=data.get("blocked", False),
            confidence=data.get("confidence", 0.0),
            obfuscated_words=obfuscated_words,
            message=data.get("message", "")
        )

    # to_dict() and to_json() inherited from BaseResponse


# ==================== Toxicity Detector API Client ====================

class ToxicityDetectorAPI(BaseRapidAPI):
    """
    Async client for Toxicity Detector API via RapidAPI.

    This API detects toxic content in Arabic and English text and audio.
    Features ML-based obfuscation detection with no static profanity lists.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with ToxicityDetectorAPI(api_key="your_key") as client:
            # Analyze text
            result = await client.analyze_text("نص للتحليل")
            print(f"Blocked: {result.blocked}")
            print(f"Confidence: {result.confidence}")

            # Analyze audio
            audio_result = await client.analyze_audio("path/to/audio.mp3")
            print(f"Transcribed: {audio_result.text}")
            print(f"Audio blocked: {audio_result.blocked}")

            # Use with config
            config = APIConfig(api_key="key", max_retries=5)
            async with ToxicityDetectorAPI(config=config) as client:
                result = await client.analyze_text("text")
    """

    BASE_URL = "https://toxicity-detector1.p.rapidapi.com"
    DEFAULT_HOST = "toxicity-detector1.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers inherited from BaseRapidAPI

    @with_retry(max_attempts=3, delay=1.0)
    async def analyze_text(self, text: str) -> TextAnalysisResult:
        """
        Analyze text for toxicity.

        Args:
            text: Text to analyze (Arabic or English)

        Returns:
            TextAnalysisResult with toxicity classification and obfuscated words

        Raises:
            ToxicityInvalidInputError: If text is empty or invalid
            ToxicityRequestError: If request fails

        Example:
            result = await client.analyze_text("نص للتحليل")
            if result.blocked:
                print(f"Blocked with {result.confidence*100}% confidence")
                for word in result.obfuscated_words:
                    print(f"  - {word.original} -> {word.corrected}")
        """
        if not text or not text.strip():
            raise InvalidInputError("Text cannot be empty")

        logger.info(f"Analyzing text: {text}...")

        payload: Dict[str, str] = {"text": text}
        data: Dict[str, Any] = await self._make_request("POST", "/analyze-text", json=payload)

        result: TextAnalysisResult = TextAnalysisResult.from_dict(data)
        logger.info(f"Text analysis complete: blocked={result.blocked}, confidence={result.confidence:.2f}")
        return result

    @with_retry(max_attempts=3, delay=1.0)
    async def analyze_audio(
        self,
        audio_path: Union[str, Path]
    ) -> AudioAnalysisResult:
        """
        Analyze audio file for toxicity.

        Supported formats: mp3, wav, m4a, ogg, flac, webm.
        The API auto-detects language (Arabic or English).

        Args:
            audio_path: Path to audio file

        Returns:
            AudioAnalysisResult with transcribed text and toxicity classification

        Raises:
            ToxicityInvalidInputError: If file doesn't exist
            ToxicityRequestError: If request fails

        Example:
            result = await client.analyze_audio("audio.mp3")
            print(f"Transcribed: {result.text}")
            if result.blocked:
                print(f"Audio blocked with {result.confidence*100}% confidence")
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise InvalidInputError(f"Audio file not found: {audio_path}")

        logger.info(f"Analyzing audio: {audio_path}")

        # Prepare multipart form data
        form = aiohttp.FormData()
        async with aiofiles.open(audio_path, 'rb') as f:
            file_content: bytes = await f.read()
            form.add_field(
                'audio',
                file_content,
                filename=audio_path.name,
                content_type='audio/mpeg'
            )

        data: Dict[str, Any] = await self._post_form_data("/analyze-audio", form)

        result: AudioAnalysisResult = AudioAnalysisResult.from_dict(data)
        logger.info(f"Audio analysis complete: blocked={result.blocked}")
        return result

    async def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information and available endpoints.

        Returns:
            Dictionary with API info and endpoints

        Example:
            info = await client.get_api_info()
            print(f"API: {info['name']} v{info['version']}")
        """
        data: Dict[str, Any] = await self._make_request("GET", "/")
        logger.info("API info retrieved")
        return data
