"""
Ayyad APIs - Collection of Python wrappers for various APIs
"""

__version__ = "0.1.8"

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
    Channel,
    Video,
    VideoInfoResponse,
    TelegramResponse,
    TelegramInfoResponse,
    DownloadResult,
    LiveStream,
    ServerDownloadField,
    ServerResponse,
    HostDownloadField,
    HostResponse,
)

from .youtube_suggest import (
    YouTubeSuggestAPI,
    SuggestError,
    APIResponseError as SuggestAPIResponseError,
    ProcessingError,
    SuggestionResult,
)

from .pinterest import (
    download_file,
    PinterestAPI,
    PinterestAPIError,
    PinterestAuthenticationError,
    PinterestDownloadError,
    PinterestInvalidURLError,
    PinterestRequestError,
    Thumbnail,
    ImageDownloadResult,
    VideoDownloadResult,
)

from .alltube_extractor import (
    AllTubeAPI,
    AllTubeError,
    AllTubeAuthenticationError,
    AllTubeRequestError,
    AllTubeInvalidURLError,
    VideoInfo,
    Format,
    Subtitle,
)

__all__ = [
    "__version__",

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
    "Channel",
    "Video",
    "VideoInfoResponse",
    "TelegramResponse",
    "TelegramInfoResponse",
    "DownloadResult",
    "LiveStream",
    "ServerDownloadField",
    "ServerResponse",
    "HostDownloadField",
    "HostResponse",

    # YouTube Suggest
    "YouTubeSuggestAPI",
    "SuggestError",
    "SuggestAPIResponseError",
    "ProcessingError",
    "SuggestionResult",

    # Pinterest
    "download_file",
    "PinterestAPI",
    "PinterestAPIError",
    "PinterestAuthenticationError",
    "PinterestDownloadError",
    "PinterestInvalidURLError",
    "PinterestRequestError",
    "Thumbnail",
    "ImageDownloadResult",
    "VideoDownloadResult",

    # AllTube CDN Extractor
    "AllTubeAPI",
    "AllTubeError",
    "AllTubeAuthenticationError",
    "AllTubeRequestError",
    "AllTubeInvalidURLError",
    "VideoInfo",
    "Format",
    "Subtitle",
]