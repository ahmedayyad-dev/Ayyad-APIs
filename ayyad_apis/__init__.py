"""
Ayyad APIs - Collection of async Python wrappers for Ahmed Ayyad's public APIs.

Seven clients are exposed:

- :class:`YouTubeAPI` — YouTube metadata, Telegram upload, Google Photos upload, server download, live HLS
- :class:`YouTubeStreamAPI` — RapidAPI YouTube streaming, info, search, and MP3/video download
- :class:`PornDetectionAPI` — NSFW detection for images and videos (porn-detection-api)
- :class:`NSFWModerationAPI` — NSFW detection, moderation, directory audit & quarantine (nsfw-image-video-moderation-api)
- :class:`TubeRelayAPI` — Direct key-in-path YouTube info, search and streaming
- :class:`AnghamiAPI` — Anghami catalog search, song metadata, 1-line downloads, and CDN streams
- :class:`TikTokAPI` — TikTok link resolution, bulk downloads, video details, stream extraction, profiles, and analytics

All clients are asynchronous and support ``async with``::

    async with YouTubeStreamAPI(api_key="...") as client:
        info = await client.get_info("dQw4w9WgXcQ")
        print(info.title, info.duration_string)
"""

__version__ = "0.8.0"

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

from .nsfw_moderation import (
    NSFWModerationAPI,
    NSFWImageVideoModerationAPI,
    NSFWModerationError,
    DirectoryScanResult,
    QuarantineResult,
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
    GoogleResponse,
    strip_thumbnail_from_video,
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

from .youtube_stream import (
    YouTubeStreamAPI,
    YouTubeStreamMP3API,
    YouTubeStreamError,
)

from .anghami import (
    AnghamiAPI,
    SongSummary,
    SongDetail,
    SearchResponse,
    AnghamiStreamUrl,
)

from .tiktok import (
    TikTokAPI,
    TikTokDataAPI,
    TikTokResolvedLink,
    TikTokShareLink,
    TikTokVideoStatus,
    TikTokUser,
    TikTokUserProfile,
    TikTokStatistics,
    TikTokMusic,
    TikTokVideoInfo,
    TikTokVideo,
    TikTokPostsResponse,
    TikTokMusicVideosResponse,
    TikTokRelatedMusicResponse,
    TikTokUserAnalytics,
    TikTokBatchDownloadResult,
    clean_tiktok_url,
    extract_aweme_id,
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

    # NSFW Moderation
    "NSFWModerationAPI",
    "NSFWImageVideoModerationAPI",
    "NSFWModerationError",
    "DirectoryScanResult",
    "QuarantineResult",

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
    "GoogleResponse",
    "strip_thumbnail_from_video",
    "DownloadResult",
    "LiveStream",
    "ServerResponse",
    "VideoSearchResult",

    # TubeRelay
    "TubeRelayAPI",
    "VideoInfo",
    "StreamUrl",
    "TubeRelayError",

    # YouTube Stream (RapidAPI)
    "YouTubeStreamAPI",
    "YouTubeStreamMP3API",
    "YouTubeStreamError",

    # Anghami
    "AnghamiAPI",
    "SongSummary",
    "SongDetail",
    "SearchResponse",
    "AnghamiStreamUrl",

    # TikTok
    "TikTokAPI",
    "TikTokDataAPI",
    "TikTokResolvedLink",
    "TikTokShareLink",
    "TikTokVideoStatus",
    "TikTokUser",
    "TikTokUserProfile",
    "TikTokStatistics",
    "TikTokMusic",
    "TikTokVideoInfo",
    "TikTokVideo",
    "TikTokPostsResponse",
    "TikTokMusicVideosResponse",
    "TikTokRelatedMusicResponse",
    "TikTokUserAnalytics",
    "TikTokBatchDownloadResult",
    "clean_tiktok_url",
    "extract_aweme_id",
]
