"""
TubeRelay API wrapper for YouTube download, info, and search.

This module provides a simple async interface to interact with TubeRelay API,
allowing users to get video info, search YouTube, stream and download media.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from urllib.parse import urlencode

import aiohttp

from ..utils import (
    BaseResponse,
    RequestError,
    DownloadError,
    get_session,
    download_file as _download_file,
)

logger = logging.getLogger(__name__)


# ==================== Exception ====================

class TubeRelayError(RequestError):
    """TubeRelay API specific error."""
    pass


# ==================== Data Model ====================

@dataclass
class VideoInfo(BaseResponse):
    """Unified video info returned by /info and /search endpoints."""
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
    _api: Optional["TubeRelayAPI"] = field(default=None, repr=False)

    def get_stream_url(self, type: str = "audio", quality: str = "best") -> str:
        """Get the stream URL for this video."""
        if not self._api:
            raise TubeRelayError("API instance not available")
        return self._api.stream(self.id, type=type, quality=quality)

    async def download(
        self,
        file_path: str,
        type: str = "audio",
        quality: str = "best",
        max_retries: int = 3,
    ) -> str:
        """Download this video/audio to a local file."""
        if not self._api:
            raise TubeRelayError("API instance not available")
        return await self._api.download(
            self.id, file_path, type=type, quality=quality, max_retries=max_retries
        )


# ==================== API Client ====================

class TubeRelayAPI:
    """
    Client for TubeRelay YouTube API.

    Example:
        async with TubeRelayAPI(api_key="your-key") as client:
            info = await client.get_info("dQw4w9WgXcQ")
            print(info.title)

            results = await client.search("python tutorial", limit=5)
            for video in results:
                print(video.title)

            url = client.stream("dQw4w9WgXcQ", type="audio", quality="128k")
            print(url)

            await client.download("dQw4w9WgXcQ", "song.mp3", type="audio")
    """

    BASE_URL = "https://tuberelay.api.ahmedayyad.dev"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "TubeRelayAPI":
        self._session = get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._session = None
        return False

    def _url(self, endpoint: str, params: Optional[dict] = None) -> str:
        """Build full URL for an endpoint."""
        path = f"{self.base_url}/{self.api_key}/{endpoint}"
        if params:
            return f"{path}?{urlencode(params)}"
        return path

    async def _get_json(self, endpoint: str, params: Optional[dict] = None) -> any:
        """Make GET request and return parsed JSON."""
        if not self._session:
            raise TubeRelayError("Session not initialized. Use async context manager.")

        url = self._url(endpoint, params)
        logger.debug(f"GET {url}")

        async with self._session.get(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise TubeRelayError(
                    f"TubeRelay API error: {resp.status} — {text}",
                    status_code=resp.status,
                    endpoint=endpoint,
                )
            return await resp.json()

    @staticmethod
    def _parse_video_info(data: dict, api: Optional["TubeRelayAPI"] = None) -> VideoInfo:
        """Parse API response dict into VideoInfo."""
        return VideoInfo(
            id=data.get("id", ""),
            title=data.get("title", ""),
            url=data.get("url", ""),
            duration=data.get("duration", 0),
            duration_string=data.get("duration_string", ""),
            thumbnail=data.get("thumbnail", ""),
            channel=data.get("channel", ""),
            channel_id=data.get("channel_id", ""),
            views=data.get("views", 0),
            views_text=data.get("views_text", ""),
            is_live=data.get("is_live", False),
            description=data.get("description"),
            keywords=data.get("keywords"),
            published=data.get("published"),
            _api=api,
        )

    # ==================== Public Methods ====================

    async def get_info(self, video_id: str) -> VideoInfo:
        """
        Get video metadata for a given YouTube video ID.

        Args:
            video_id: YouTube video ID (11-12 characters)

        Returns:
            VideoInfo with title, duration, thumbnail, channel, views, etc.
        """
        data = await self._get_json("info", {"video_id": video_id})
        return self._parse_video_info(data, api=self)

    async def search(self, query: str, limit: int = 10) -> List[VideoInfo]:
        """
        Search YouTube or resolve a YouTube URL to video metadata.

        Args:
            query: Search text or a YouTube URL
            limit: Maximum results to return (1-50)

        Returns:
            List of VideoInfo objects
        """
        data = await self._get_json("search", {"query": query, "limit": limit})
        if not isinstance(data, list):
            return []
        return [self._parse_video_info(item, api=self) for item in data]

    def stream(
        self,
        video_id: str,
        type: str = "audio",
        quality: str = "best",
    ) -> str:
        """
        Get the stream URL for a YouTube video (no HTTP request).

        Args:
            video_id: YouTube video ID (11-12 characters)
            type: "audio" or "video"
            quality: "best", "worst", "128k" (audio), "720p" (video), etc.

        Returns:
            Direct stream URL string
        """
        return self._url("stream", {
            "video_id": video_id,
            "type": type,
            "quality": quality,
        })

    async def download(
        self,
        video_id: str,
        file_path: str,
        type: str = "audio",
        quality: str = "best",
        max_retries: int = 3,
    ) -> str:
        """
        Download audio or video from YouTube to a local file.

        Args:
            video_id: YouTube video ID (11-12 characters)
            file_path: Output file path
            type: "audio" or "video"
            quality: "best", "worst", "128k" (audio), "720p" (video), etc.
            max_retries: Download retry attempts (default 3)

        Returns:
            Saved file path
        """
        url = self.stream(video_id, type=type, quality=quality)

        result_path = await _download_file(
            url=url,
            output_path=file_path,
            max_retries=max_retries,
            session=self._session,
        )
        if result_path is None:
            raise TubeRelayError(f"Failed to download {video_id} to {file_path}")
        return result_path
