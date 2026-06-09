"""
Type definitions for Ayyad APIs.

This module provides type hints, literals, and TypedDict definitions
for better type safety and IDE support across all API modules.
"""

from typing import Literal, TypedDict, Union, List, Dict, Any, Callable
try:
    from typing_extensions import NotRequired
except ImportError:
    # For Python 3.11+, NotRequired is in typing
    from typing import NotRequired  # type: ignore


# ==================== Common Literals ====================

HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
"""HTTP methods supported by API clients."""

Quality = Literal["original", "high", "medium", "low"]
"""Image quality levels for Pinterest and other media APIs."""

MediaType = Literal["image", "video", "auto"]
"""Media type for download operations."""

PornLabel = Literal["Safe", "Unsafe"]
"""Porn detection labels."""

Language = Literal["ar", "en", "fr", "es", "de", "it", "pt", "ru", "zh", "ja", "ko"]
"""Supported language codes."""

DeepSeekModelType = Literal["default", "expert", "vision"]
"""DeepSeek model types for ZulvexAI."""


# ==================== TypedDict Definitions ====================


class RapidAPIResponse(TypedDict):
    """Standard RapidAPI response structure."""
    success: bool
    data: NotRequired[Dict[str, Any]]
    error: NotRequired[str]
    message: NotRequired[str]


class VideoMetadataDict(TypedDict):
    """Video metadata type."""
    duration: NotRequired[int]
    width: NotRequired[int]
    height: NotRequired[int]
    format: NotRequired[str]
    bitrate: NotRequired[int]
    fps: NotRequired[float]


class ErrorDict(TypedDict):
    """Error information dictionary."""
    error_type: str
    message: str
    status_code: NotRequired[int]
    endpoint: NotRequired[str]
    retry_count: NotRequired[int]
    timestamp: float


# ==================== Type Aliases ====================

# Import ProgressInfo for type alias
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .utils import ProgressInfo

ProgressCallback = Callable[["ProgressInfo"], None]
"""Callback function for progress tracking."""

DownloadResult = Union[bytes, str, None]
"""Result of download operation: bytes, file path string, or None if failed."""

JsonDict = Dict[str, Any]
"""JSON dictionary type."""

Headers = Dict[str, str]
"""HTTP headers dictionary."""


# ==================== API-Specific Types ====================


class PornDetectionResultDict(TypedDict):
    """Porn detection result structure."""
    label: PornLabel
    confidence: float
    is_safe: bool


class YouTubeVideoDict(TypedDict):
    """YouTube video information."""
    video_id: str
    title: str
    duration: NotRequired[int]
    thumbnail: NotRequired[str]
    channel: NotRequired[str]


# ==================== Exports ====================

__all__ = [
    # Literals
    "HttpMethod",
    "Quality",
    "MediaType",
    "PornLabel",
    "Language",

    # TypedDict
    "RapidAPIResponse",
    "VideoMetadataDict",
    "ErrorDict",
    "PornDetectionResultDict",
    "YouTubeVideoDict",

    # Type Aliases
    "ProgressCallback",
    "DownloadResult",
    "JsonDict",
    "Headers",
]
