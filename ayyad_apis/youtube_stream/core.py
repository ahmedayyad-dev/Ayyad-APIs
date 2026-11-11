"""
YouTube Stream MP3 URL API wrapper.

This module provides an async interface to the YouTube Stream MP3 URL API
hosted on RapidAPI (youtube-stream-mp3-url-api.p.rapidapi.com). It allows
users to query YouTube video metadata, search YouTube, resolve direct
temporary playback/stream URLs, and download audio or video directly.

Author: Ahmed Ayyad
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp

from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    InvalidInputError,
    RequestError,
    download_file as _download_file,
    with_retry,
)

logger = logging.getLogger(__name__)

_YOUTUBE_ID_PATTERN = re.compile(r"(?:v=|\/|youtu\.be\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})")


class YouTubeStreamError(RequestError):
    """Exception raised for YouTube Stream API errors."""
    pass


@dataclass
class StreamUrl(BaseResponse):
    """Temporary playback URL returned by ``/stream-url``."""

    url: str = ""
    expires_in: int = 300
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class VideoInfo(BaseResponse):
    """Video details returned by ``/info`` and ``/search``."""

    id: str = ""
    title: str = ""
    url: str = ""
    duration: int = 0
    duration_string: str = ""
    thumbnail: str = ""
    channel: str = ""
    channel_id: str = ""
    views: int = 0
    views_text: str = ""
    is_live: bool = False
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    published: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    _api: Optional["YouTubeStreamAPI"] = field(default=None, repr=False)

    async def get_stream_url(
        self,
        type: str = "audio",
        quality: str = "best",
    ) -> StreamUrl:
        """Get a temporary direct CDN stream URL for this video."""
        if not self._api:
            raise YouTubeStreamError("API instance not bound to this VideoInfo object.")
        return await self._api.stream_url(self.id, type=type, quality=quality)

    async def download(
        self,
        file_path: Optional[Union[str, Path]] = None,
        type: str = "audio",
        quality: str = "best",
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """Download this video or audio directly."""
        if not self._api:
            raise YouTubeStreamError("API instance not bound to this VideoInfo object.")
        return await self._api.download(
            self.id,
            file_path=file_path,
            type=type,
            quality=quality,
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session,
        )


class YouTubeStreamAPI(BaseRapidAPI):
    """
    Async client for YouTube Stream MP3 URL API hosted on RapidAPI.

    API Host: ``youtube-stream-mp3-url-api.p.rapidapi.com``

    Supports:
    - Video metadata extraction (``get_info``)
    - YouTube search (``search``)
    - Temporary CDN playback URL generation (``stream_url``)
    - Direct media download (``download``)
    - One-line search-and-download shortcut (``quick_download``)

    Accepts bare 11-char video IDs or full YouTube URLs (e.g. ``youtu.be``,
    ``youtube.com/watch?v=...``, ``shorts/...``) transparently.

    Example:
        async with YouTubeStreamAPI(api_key="your_key") as api:
            info = await api.get_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            print(info.title, info.duration_string)

            # Get temporary direct stream URL
            stream = await api.stream_url(info.id, type="audio")
            print(stream.url, stream.expires_in)

            # Download audio MP3
            await api.download(info.id, "rick.mp3", type="audio")
    """

    BASE_URL = "https://youtube-stream-mp3-url-api.p.rapidapi.com"
    DEFAULT_HOST = "youtube-stream-mp3-url-api.p.rapidapi.com"

    @staticmethod
    def _extract_video_id(video_id_or_url: str) -> str:
        """Extract 11-character YouTube video ID from a URL or raw ID string."""
        if not video_id_or_url or not str(video_id_or_url).strip():
            raise InvalidInputError("Video ID or URL cannot be empty.")

        val = str(video_id_or_url).strip()
        if len(val) == 11 and "/" not in val and "?" not in val:
            return val

        match = _YOUTUBE_ID_PATTERN.search(val)
        if match:
            return match.group(1)

        # Fallback to the trimmed value
        return val

    def _parse_video_info(self, data: Dict[str, Any]) -> VideoInfo:
        """Parse raw dictionary into VideoInfo instance."""
        return VideoInfo(
            id=str(data.get("id") or ""),
            title=data.get("title") or "",
            url=data.get("url") or "",
            duration=int(data.get("duration") or 0),
            duration_string=data.get("duration_string") or "",
            thumbnail=data.get("thumbnail") or "",
            channel=data.get("channel") or "",
            channel_id=data.get("channel_id") or "",
            views=int(data.get("views") or 0),
            views_text=data.get("views_text") or "",
            is_live=bool(data.get("is_live")),
            description=data.get("description"),
            keywords=data.get("keywords"),
            published=data.get("published"),
            raw=data,
            _api=self,
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def get_info(self, video_id_or_url: str) -> VideoInfo:
        """
        Fetch video metadata from YouTube.

        Args:
            video_id_or_url: 11-character video ID or full YouTube URL.

        Returns:
            VideoInfo instance with title, duration, channel, and thumbnail.

        Raises:
            InvalidInputError: If video ID cannot be parsed.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server failures.
        """
        video_id = self._extract_video_id(video_id_or_url)
        data = await self._make_request("GET", "/info", params={"video_id": video_id})
        if not isinstance(data, dict):
            raise YouTubeStreamError("Unexpected response shape from /info", endpoint="/info")
        return self._parse_video_info(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, limit: int = 10) -> List[VideoInfo]:
        """
        Search YouTube for videos.

        Args:
            query: Search query string.
            limit: Maximum number of results to return (default: 10).

        Returns:
            List of VideoInfo instances.

        Raises:
            InvalidInputError: If query is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server failures.
        """
        q = (query or "").strip()
        if not q:
            raise InvalidInputError("Search query cannot be empty.")

        clamped_limit = max(1, min(50, int(limit)))
        data = await self._make_request("GET", "/search", params={"query": q, "limit": clamped_limit})
        if not isinstance(data, list):
            raise YouTubeStreamError("Unexpected response shape from /search", endpoint="/search")

        return [self._parse_video_info(item) for item in data if isinstance(item, dict)]

    @with_retry(max_attempts=3, delay=1.0)
    async def stream_url(
        self,
        video_id_or_url: str,
        type: str = "audio",
        quality: str = "best",
    ) -> StreamUrl:
        """
        Get a key-free, temporary CDN stream URL for a video.

        Args:
            video_id_or_url: 11-character video ID or full YouTube URL.
            type: Media stream type (``"audio"`` or ``"video"``, default: ``"audio"``).
            quality: Desired quality (``"best"``, ``"128k"``, ``"192k"``, ``"320k"``, etc.).

        Returns:
            StreamUrl containing direct temporary URL and expiration seconds.

        Raises:
            InvalidInputError: If video ID is invalid or type is unrecognized.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server failures.
        """
        video_id = self._extract_video_id(video_id_or_url)
        t = type.lower().strip()
        if t not in ("audio", "video"):
            raise InvalidInputError(f"Invalid type: {type!r}. Must be 'audio' or 'video'.")

        params = {"video_id": video_id, "type": t, "quality": quality}
        data = await self._make_request("GET", "/stream-url", params=params)
        if not isinstance(data, dict) or "url" not in data:
            raise YouTubeStreamError("Unexpected response shape from /stream-url", endpoint="/stream-url")

        return StreamUrl(
            url=data["url"],
            expires_in=int(data.get("expires_in") or 300),
            raw=data,
        )

    async def download(
        self,
        video_id_or_url: str,
        file_path: Optional[Union[str, Path]] = None,
        type: str = "audio",
        quality: str = "best",
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Resolve a stream URL and download media directly to disk.

        Args:
            video_id_or_url: 11-character video ID or full YouTube URL.
            file_path: Destination file path. If None, defaults to ``<video_id>.<ext>``.
            type: Media stream type (``"audio"`` or ``"video"``).
            quality: Desired quality (``"best"``, ``"128k"``, etc.).
            show_progress: Whether to log download progress.
            progress_callback: Optional callback receiving (bytes_downloaded, total_bytes).
            session: Optional aiohttp ClientSession.

        Returns:
            Path string to the saved file.
        """
        video_id = self._extract_video_id(video_id_or_url)
        stream_info = await self.stream_url(video_id, type=type, quality=quality)

        default_ext = ".mp3" if type == "audio" else ".mp4"
        return await _download_file(
            stream_info.url,
            output_path=file_path,
            default_filename=f"yt_{video_id}",
            default_ext=default_ext,
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session or self._session,
        )

    async def quick_download(
        self,
        query_or_url: str,
        output_path: Optional[Union[str, Path]] = None,
        type: str = "audio",
        quality: str = "best",
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        One-liner search and download helper.

        If a URL or video ID is provided, it downloads it directly.
        If a text search query is provided, it searches YouTube, selects the top match,
        and downloads it.

        Args:
            query_or_url: Video URL, video ID, or search query text.
            output_path: Optional output file destination.
            type: Media stream type (``"audio"`` or ``"video"``).
            quality: Desired quality.
            show_progress: Whether to log download progress.
            progress_callback: Optional progress callback.

        Returns:
            Path string to the saved file.
        """
        target = query_or_url.strip()
        if target.startswith("http://") or target.startswith("https://") or (len(target) == 11 and " " not in target):
            video_id = target
        else:
            results = await self.search(target, limit=1)
            if not results:
                raise InvalidInputError(f"No YouTube videos found matching: {target!r}")
            video_id = results[0].id

        return await self.download(
            video_id,
            file_path=output_path,
            type=type,
            quality=quality,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
