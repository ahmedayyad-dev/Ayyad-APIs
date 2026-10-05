"""
Ayyad APIs - Collection of async Python wrappers for Ahmed Ayyad's public APIs.

Three clients are exposed:

- :class:`YouTubeAPI` — YouTube metadata, Telegram upload, server download, live HLS
- :class:`PornDetectionAPI` — NSFW detection for images and videos
- :class:`TubeRelayAPI` — YouTube info, search and streaming

All clients are asynchronous and support ``async with``::

    async with YouTubeAPI(api_key="...") as client:
        print(await client.video_info("https://youtu.be/dQw4w9WgXcQ"))
"""

__version__ = "0.4.0"

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
    ClientError,
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
    VideoAnalysis,
    VideoSegment,
    UploadUrl,
)

from .youtube_to_telegram import (
    YouTubeAPI,
    APIResponseError as YouTubeAPIResponseError,
    DownloadInProgressError,
    QueuedJobResponse,
    Channel,
    Video,
    VideoInfoResponse,
    TelegramResponse,
    DownloadResult,
    LiveStream,
    ServerResponse,
    VideoSearchResult,
    JobStatus,
)

from .tube_relay import (
    TubeRelayAPI,
    VideoInfo,
    StreamUrl,
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
    "VideoAnalysis",
    "VideoSegment",
    "UploadUrl",

    # YouTube to Telegram
    "YouTubeAPI",
    "YouTubeAPIResponseError",
    "DownloadInProgressError",
    "QueuedJobResponse",
    "JobStatus",
    "Channel",
    "Video",
    "VideoInfoResponse",
    "TelegramResponse",
    "DownloadResult",
    "LiveStream",
    "ServerResponse",
    "VideoSearchResult",

    # TubeRelay
    "TubeRelayAPI",
    "VideoInfo",
    "StreamUrl",
    "TubeRelayError",
]