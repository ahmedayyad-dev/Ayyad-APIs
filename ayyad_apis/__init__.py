"""
Ayyad APIs - Collection of Python wrappers for various APIs
"""

__version__ = "0.2.7"

# Import shared utilities
from .utils import (
    download_file,
    create_rapidapi_headers,
    validate_rapidapi_response,
    # Base classes
    BaseResponse,
    BaseRapidAPI,
    # Exception hierarchy
    APIError,
    AuthenticationError,
    ClientError,
    RequestError,
    InvalidInputError,
    DownloadError as DownloadError,
    # Configuration
    APIConfig,
    # Progress tracking
    ProgressTracker,
    ProgressInfo,
    # Decorators
    with_retry,
)

# Expose main APIs at the package root
from .porn_detection import (
    PornDetectionAPI,
    DetectionError as PornDetectionError,
    APIResponseError as PornAPIResponseError,
    UploadError,
    VideoAnalysisConfig,
    ImageDetectionResult,
    VideoDetectionResult,
    VideoStats,
    VideoThresholds,
    UploadUrl,
)

from .youtube_to_telegram import (
    YouTubeAPI,
    DownloadError as YouTubeDownloadError,
    APIResponseError as YouTubeAPIResponseError,
    Channel,
    Video,
    VideoInfoResponse,
    TelegramResponse,
    DownloadResult,
    LiveStream,
    ServerResponse,
    # Progress tracking (2026-01-19)
    DownloadProgress,
    BackgroundJobResponse,
    # Deprecated (v26.01.21+) - kept for backward compatibility
    ServerDownloadField,
)

from .youtube_suggest import (
    YouTubeSuggestAPI,
    SuggestError,
    APIResponseError as SuggestAPIResponseError,
    ProcessingError,
    SuggestionResult,
)

from .pinterest import (
    PinterestAPI,
    PinterestAPIError,
    PinterestAuthenticationError,
    PinterestClientError,
    PinterestDownloadError,
    PinterestInvalidURLError,
    PinterestRequestError,
    Thumbnail,
    ImageMetadata,
    ImageDownloadResult,
    VideoDownloadResult,
    BoardDownloadResult,
    BatchDownloadResult,
    ProfileDownloadResult,
)

from .alltube_extractor import (
    AllTubeAPI,
    AllTubeError,
    AllTubeAuthenticationError,
    AllTubeClientError,
    AllTubeRequestError,
    AllTubeInvalidURLError,
)

from .toxicity_detector import (
    ToxicityDetectorAPI,
    ToxicityDetectorError,
    ToxicityAuthenticationError,
    ToxicityClientError,
    ToxicityRequestError,
    ToxicityInvalidInputError,
    ObfuscatedWord,
    TextAnalysisResult,
    AudioAnalysisResult,
)

from .translator import (
    TranslatorAPI,
    TranslatorError,
    TranslatorAuthenticationError,
    TranslatorClientError,
    TranslatorRequestError,
    TranslatorInvalidInputError,
    TranslationResult,
)

__all__ = [
    "__version__",

    # Shared Utilities
    "download_file",
    "create_rapidapi_headers",
    "validate_rapidapi_response",

    # Base Classes
    "BaseResponse",
    "BaseRapidAPI",

    # Exception Hierarchy
    "APIError",
    "AuthenticationError",
    "ClientError",
    "RequestError",
    "InvalidInputError",
    "DownloadError",

    # Configuration
    "APIConfig",

    # Progress Tracking
    "ProgressTracker",
    "ProgressInfo",

    # Decorators
    "with_retry",

    # Porn Detection
    "PornDetectionAPI",
    "PornDetectionError",
    "PornAPIResponseError",
    "UploadError",
    "VideoAnalysisConfig",
    "ImageDetectionResult",
    "VideoDetectionResult",
    "VideoStats",
    "VideoThresholds",
    "UploadUrl",

    # YouTube to Telegram
    "YouTubeAPI",
    "YouTubeDownloadError",
    "YouTubeAPIResponseError",
    "Channel",
    "Video",
    "VideoInfoResponse",
    "TelegramResponse",
    "DownloadResult",
    "LiveStream",
    "ServerDownloadField",
    "ServerResponse",
    "DownloadProgress",
    "BackgroundJobResponse",

    # YouTube Suggest
    "YouTubeSuggestAPI",
    "SuggestError",
    "SuggestAPIResponseError",
    "ProcessingError",
    "SuggestionResult",

    # Pinterest
    "PinterestAPI",
    "PinterestAPIError",
    "PinterestAuthenticationError",
    "PinterestClientError",
    "PinterestDownloadError",
    "PinterestInvalidURLError",
    "PinterestRequestError",
    "Thumbnail",
    "ImageMetadata",
    "ImageDownloadResult",
    "VideoDownloadResult",
    "BoardDownloadResult",
    "BatchDownloadResult",
    "ProfileDownloadResult",

    # AllTube CDN Extractor
    "AllTubeAPI",
    "AllTubeError",
    "AllTubeAuthenticationError",
    "AllTubeClientError",
    "AllTubeRequestError",
    "AllTubeInvalidURLError",

    # Toxicity Detector
    "ToxicityDetectorAPI",
    "ToxicityDetectorError",
    "ToxicityAuthenticationError",
    "ToxicityClientError",
    "ToxicityRequestError",
    "ToxicityInvalidInputError",
    "ObfuscatedWord",
    "TextAnalysisResult",
    "AudioAnalysisResult",

    # AI Translator
    "TranslatorAPI",
    "TranslatorError",
    "TranslatorAuthenticationError",
    "TranslatorClientError",
    "TranslatorRequestError",
    "TranslatorInvalidInputError",
    "TranslationResult",
]