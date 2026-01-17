"""
Toxicity Detector API wrapper for detecting toxic content in text and audio.

This module provides a simple async interface to interact with Toxicity Detector API
through RapidAPI, allowing users to detect toxic language in Arabic and English.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    RequestError,
    InvalidInputError,
    APIConfig,
    with_retry,
)

# Configure logging
logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

# Create aliases for backward compatibility
ToxicityDetectorError = APIError
ToxicityAuthenticationError = AuthenticationError
ToxicityRequestError = RequestError
ToxicityInvalidInputError = InvalidInputError


# ==================== Data Models ====================

@dataclass
class BlockedWord(BaseResponse):
    """Represents a blocked/toxic word with obfuscation details."""
    obfuscated: str
    clean: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlockedWord":
        """Create BlockedWord from API response dictionary."""
        return cls(
            obfuscated=data.get("obfuscated", ""),
            clean=data.get("clean", "")
        )

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class TextAnalysisResult(BaseResponse):
    """Result from text toxicity analysis."""
    confidence: float
    is_toxic: bool
    words: List[BlockedWord]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextAnalysisResult":
        """Create TextAnalysisResult from API response dictionary."""
        words = []
        if "words" in data and isinstance(data["words"], list):
            words = [BlockedWord.from_dict(w) for w in data["words"]]

        return cls(
            confidence=data.get("confidence", 0.0),
            is_toxic=data.get("is_toxic", False),
            words=words
        )

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class AudioAnalysisResult(BaseResponse):
    """Result from audio toxicity analysis."""
    success: bool
    message: Optional[str] = None
    confidence: Optional[float] = None
    is_toxic: Optional[bool] = None
    words: List[BlockedWord] = None

    def __post_init__(self):
        if self.words is None:
            self.words = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioAnalysisResult":
        """Create AudioAnalysisResult from API response dictionary."""
        words = []
        if "words" in data and isinstance(data["words"], list):
            words = [BlockedWord.from_dict(w) for w in data["words"]]

        return cls(
            success=data.get("success", False),
            message=data.get("message"),
            confidence=data.get("confidence"),
            is_toxic=data.get("is_toxic"),
            words=words
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
            print(f"Is toxic: {result.is_toxic}")
            print(f"Confidence: {result.confidence}")

            # Analyze audio
            audio_result = await client.analyze_audio("path/to/audio.mp3")
            print(f"Audio is toxic: {audio_result.is_toxic}")

            # Use with config
            config = APIConfig(api_key="key", max_retries=5)
            async with ToxicityDetectorAPI(config=config) as client:
                result = await client.analyze_text("text")
    """

    BASE_URL = "https://toxicity-detector1.p.rapidapi.com"
    DEFAULT_HOST = "toxicity-detector1.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers inherited from BaseRapidAPI

    async def _make_request_with_file(
        self,
        endpoint: str,
        form_data: "aiohttp.FormData"
    ) -> Dict[str, Any]:
        """
        Make an async request with file upload (for audio analysis).

        Args:
            endpoint: API endpoint path
            form_data: Form data with audio file

        Returns:
            JSON response as dictionary

        Raises:
            AuthenticationError: If authentication fails
            RequestError: If request fails
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        import aiohttp
        url = f"{self.BASE_URL}{endpoint}"

        # For file uploads, don't include Content-Type - aiohttp sets it automatically
        headers = {
            "x-rapidapi-host": self.rapidapi_host,
            "x-rapidapi-key": self.api_key
        }

        logger.debug(f"Making POST request with file to {endpoint}")

        try:
            async with self._session.post(url, headers=headers, data=form_data) as response:
                # Check for authentication errors
                if response.status in (401, 403):
                    raise AuthenticationError(
                        f"Authentication failed",
                        status_code=response.status,
                        endpoint=endpoint
                    )

                # Check for other errors
                if response.status != 200:
                    error_text = await response.text()
                    raise RequestError(
                        f"Request failed",
                        status_code=response.status,
                        response_text=error_text,
                        endpoint=endpoint
                    )

                data = await response.json()
                logger.debug("File upload request successful")
                return data

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            raise RequestError(
                f"Network error: {str(e)}",
                endpoint=endpoint,
                original_error=e
            )

    @with_retry(max_attempts=3, delay=1.0)
    async def analyze_text(self, text: str) -> TextAnalysisResult:
        """
        Analyze text for toxicity.

        Args:
            text: Text to analyze (Arabic or English)

        Returns:
            TextAnalysisResult with toxicity classification and blocked words

        Raises:
            ToxicityInvalidInputError: If text is empty or invalid
            ToxicityRequestError: If request fails

        Example:
            result = await client.analyze_text("نص للتحليل")
            if result.is_toxic:
                print(f"Toxic with {result.confidence*100}% confidence")
                for word in result.words:
                    print(f"  - {word.obfuscated} -> {word.clean}")
        """
        if not text or not text.strip():
            raise InvalidInputError("Text cannot be empty")

        logger.info(f"Analyzing text: {text[:50]}...")

        payload = {"text": text}
        data = await self._make_request("POST", "/analyze-words", json=payload)

        result = TextAnalysisResult.from_dict(data)
        logger.info(f"Text analysis complete: is_toxic={result.is_toxic}, confidence={result.confidence:.2f}")
        return result

    async def analyze_audio(
        self,
        audio_path: Union[str, Path],
        language: str = "ar"
    ) -> AudioAnalysisResult:
        """
        Analyze audio file for toxicity.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            language: Language code ('ar' for Arabic, 'en' for English)

        Returns:
            AudioAnalysisResult with toxicity classification

        Raises:
            ToxicityInvalidInputError: If file doesn't exist
            ToxicityRequestError: If request fails

        Example:
            result = await client.analyze_audio("audio.mp3", language="ar")
            if result.success and result.is_toxic:
                print(f"Audio contains toxic content: {result.confidence*100}% confidence")
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise InvalidInputError(f"Audio file not found: {audio_path}")

        logger.info(f"Analyzing audio: {audio_path}")

        # Prepare multipart form data
        import aiohttp
        form = aiohttp.FormData()
        form.add_field(
            'audio',
            open(audio_path, 'rb'),
            filename=audio_path.name,
            content_type='audio/mpeg'
        )
        form.add_field('language', language)

        data = await self._make_request_with_file("/analyze-audio", form)

        result = AudioAnalysisResult.from_dict(data)
        logger.info(f"Audio analysis complete: success={result.success}")
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

    async def model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model information

        Example:
            info = await client.model_info()
            print(f"Model info: {info}")
        """
        data = await self._make_request("GET", "/model-info")
        logger.info("Model info retrieved")
        return data
