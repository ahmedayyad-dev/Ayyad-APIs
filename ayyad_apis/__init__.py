"""
Ayyad APIs - Collection of Python wrappers for various APIs
"""

__version__ = "0.2.0"

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
    RequestError,
    InvalidInputError,
    DownloadError,
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
    DownloadError,
    APIResponseError as YouTubeAPIResponseError,
    BackgroundJobError,
    Channel,
    Video,
    VideoInfoResponse,
    TelegramResponse,
    DownloadResult,
    LiveStream,
    ServerDownloadField,
    ServerResponse,
    TryAfterResponse,
    DownloadProgressResponse,
    VideoSearchResult,
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
    AllTubeRequestError,
    AllTubeInvalidURLError,
)

from .toxicity_detector import (
    ToxicityDetectorAPI,
    ToxicityDetectorError,
    ToxicityAuthenticationError,
    ToxicityRequestError,
    ToxicityInvalidInputError,
    BlockedWord,
    TextAnalysisResult,
    AudioAnalysisResult,
)

from .translator import (
    TranslatorAPI,
    TranslatorError,
    TranslatorAuthenticationError,
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
    "DownloadError",
    "YouTubeAPIResponseError",
    "BackgroundJobError",
    "Channel",
    "Video",
    "VideoInfoResponse",
    "TelegramResponse",
    "DownloadResult",
    "LiveStream",
    "ServerDownloadField",
    "ServerResponse",
    "TryAfterResponse",
    "DownloadProgressResponse",
    "VideoSearchResult",

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
    "AllTubeRequestError",
    "AllTubeInvalidURLError",

    # Toxicity Detector
    "ToxicityDetectorAPI",
    "ToxicityDetectorError",
    "ToxicityAuthenticationError",
    "ToxicityRequestError",
    "ToxicityInvalidInputError",
    "BlockedWord",
    "TextAnalysisResult",
    "AudioAnalysisResult",

    # AI Translator
    "TranslatorAPI",
    "TranslatorError",
    "TranslatorAuthenticationError",
    "TranslatorRequestError",
    "TranslatorInvalidInputError",
    "TranslationResult",
]