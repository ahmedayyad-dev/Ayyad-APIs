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

__all__ = [
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
]