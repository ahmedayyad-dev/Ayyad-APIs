"""
Ayyad APIs - Collection of Python wrappers for various APIs
"""

__version__ = "0.2.7"

# Import shared utilities
from .utils import (
    download_file,
    create_rapidapi_headers,
    validate_rapidapi_response,
    get_session,
    close_session,
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
    JobStatus,
)

from .tube_relay import (
    TubeRelayAPI,
    VideoInfo,
    TubeRelayError,
)

__all__ = [
    "__version__",

    # Shared Utilities
    "download_file",
    "create_rapidapi_headers",
    "validate_rapidapi_response",
    "get_session",
    "close_session",

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
    "YouTubeAPIResponseError",
    "BackgroundJobError",
    "JobStatus",
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

    # TubeRelay
    "TubeRelayAPI",
    "VideoInfo",
    "TubeRelayError",
]
