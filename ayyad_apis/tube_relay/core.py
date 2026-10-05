"""
TubeRelay API wrapper for YouTube download, info, and search.

This module provides a simple async interface to interact with TubeRelay API,
allowing users to get video info, search YouTube, stream and download media.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Any
from urllib.parse import urlencode

import aiohttp

from ..utils import (
    BaseResponse,
    RequestError,
    APIConfig,
    with_retry,
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
class StreamUrl(BaseResponse):
    """Time-limited playback URL returned by ``/stream-url``.

    Unlike :attr:`VideoInfo.get_stream_url`, this URL does **not** contain the
    API key, so it is safe to hand to a third party (a player, a CDN).
    """
    url: str = ""
    expires_in: int = 300


@dataclass
class VideoInfo(BaseResponse):
    """Unified video info returned by /info and /search endpoints.

    Note:
        ``/info`` and ``/search`` do not return the same key set. ``published``
        only comes from ``/search``, while ``description`` and ``keywords`` only
        come from ``/info`` (or from a ``/search`` hit that was served out of
        the metadata cache). All three default to ``None``.

    Warning:
        ``views`` returned by ``/search`` is parsed from YouTube's abbreviated
        display text, so ``"1.2M views"`` becomes ``12``. It is a rough
        magnitude only — call :meth:`TubeRelayAPI.get_info` for an exact count.
    """
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
        """Get the stream URL for this video (contains the API key)."""
        if not self._api:
            raise TubeRelayError("API instance not available")
        return self._api.stream(self.id, type=type, quality=quality)

    async def get_public_stream_url(
        self,
        type: str = "audio",
        quality: str = "best",
    ) -> StreamUrl:
        """Get a key-free, time-limited playback URL for this video."""
        if not self._api:
            raise TubeRelayError("API instance not available")
        return await self._api.stream_url(self.id, type=type, quality=quality)

    async def download(
        self,
        file_path: str,
        type: str = "audio",
        quality: str = "best",
        max_retries: int = 3,
    ) -> str:
        """Download this video/audio to a local file.

        Warning:
            The server sends no Content-Type, no filename and no Content-Length,
            so you must supply the extension yourself. ``type="audio"`` may come
            back as raw AAC/ADTS rather than an m4a container depending on which
            internal path served the request.
        """
        if not self._api:
            raise TubeRelayError("API instance not available")
        return await self._api.download(
            self.id, file_path, type=type, quality=quality, max_retries=max_retries
        )


# ==================== API Client ====================

class TubeRelayAPI:
    """
    Client for TubeRelay YouTube API.

    Authentication is by API key **in the URL path** (``/{api_key}/{endpoint}``),
    not by header, so this client deliberately does not use ``BaseRapidAPI``.

    Warning:
        An invalid or inactive key is answered with **HTTP 307 redirect to "/"**
        (the web login page), never 401/403. Redirects are therefore disabled
        here; otherwise aiohttp would follow the redirect and hand back an HTML
        page that fails JSON parsing with a confusing error.

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
        config: Optional[APIConfig] = None,
        own_session: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or (config.rapidapi_host if config else None) or self.BASE_URL).rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=config.timeout if config else timeout)
        self.config = config
        self._own_session: bool = own_session
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "TubeRelayAPI":
        if self._own_session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        else:
            self._session = get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._own_session and self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        return False

    def _url(self, endpoint: str, params: Optional[dict] = None) -> str:
        """Build full URL for an endpoint."""
        path = f"{self.base_url}/{self.api_key}/{endpoint}"
        if params:
            return f"{path}?{urlencode(params)}"
        return path

    async def _get_json(self, endpoint: str, params: Optional[dict] = None) -> Any:
        """Make GET request and return parsed JSON.

        Raises:
            TubeRelayError: On a non-200 response, an auth redirect, or a
                non-JSON body.
        """
        if not self._session:
            raise TubeRelayError("Session not initialized. Use async context manager.")

        url = self._url(endpoint, params)
        logger.debug(f"GET {url}")

        # allow_redirects must stay False: an invalid key returns 307 -> "/".
        async with self._session.get(url, allow_redirects=False) as resp:
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location", "/")
                raise TubeRelayError(
                    f"TubeRelay authentication failed: server redirected to "
                    f"{location!r} (HTTP {resp.status}). The API key is unknown "
                    f"or inactive.",
                    status_code=resp.status,
                    endpoint=endpoint,
                    non_retryable=True,
                )

            if resp.status != 200:
                text = await resp.text()
                raise TubeRelayError(
                    f"TubeRelay API error: {resp.status} — {text}",
                    status_code=resp.status,
                    endpoint=endpoint,
                    response_text=text,
                )

            try:
                return await resp.json()
            except (aiohttp.ContentTypeError, ValueError) as e:
                text = await resp.text()
                raise TubeRelayError(
                    f"TubeRelay returned a non-JSON body: {text[:200]}",
                    status_code=resp.status,
                    endpoint=endpoint,
                    original_error=e,
                    non_retryable=True,
                ) from e

    @staticmethod
    def _validate_id(video_id: str) -> str:
        """Validate a bare YouTube ID before hitting the server.

        The server constrains ``video_id`` to 11-12 characters, so a full URL
        would come back as an opaque 422. Fail fast with a clear message.
        """
        if not video_id:
            raise TubeRelayError("video_id is required")

        if "://" in video_id or video_id.startswith("www."):
            raise TubeRelayError(
                f"video_id must be a bare YouTube ID (11-12 characters), not a "
                f"URL. Extract it from the URL, or use search() which accepts URLs."
            )

        if not (11 <= len(video_id) <= 12):
            raise TubeRelayError(
                f"video_id must be 11-12 characters, got {len(video_id)}: "
                f"{video_id!r}"
            )

        return video_id

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

    @with_retry(max_attempts=3, delay=1.0)
    async def get_info(self, video_id: str) -> VideoInfo:
        """
        Get video metadata for a given YouTube video ID.

        Args:
            video_id: YouTube video ID (11-12 characters)

        Returns:
            VideoInfo with title, duration, thumbnail, channel, views, etc.

        Raises:
            TubeRelayError: If the key is invalid, or the video is not found.
        """
        data = await self._get_json("info", {"video_id": self._validate_id(video_id)})
        return self._parse_video_info(data, api=self)

    @with_retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, limit: int = 10) -> List[VideoInfo]:
        """
        Search YouTube or resolve a YouTube URL to video metadata.

        Args:
            query: Search text, or a full YouTube URL (must include ``http://``
                or ``https://`` or it is treated as a text search).
            limit: Maximum results to return (1-50). Note the server caps the
                cached depth of a repeated query at the smallest limit ever used.

        Returns:
            List of VideoInfo objects. Empty list when there are no matches.
        """
        data = await self._get_json("search", {"query": query, "limit": limit})
        if not isinstance(data, list):
            return []
        return [self._parse_video_info(item, api=self) for item in data]

    async def stream_url(
        self,
        video_id: str,
        type: str = "audio",
        quality: str = "best",
    ) -> StreamUrl:
        """
        Get a temporary, key-free playback URL for a YouTube video.

        The returned URL expires after 300 seconds and does not contain the API
        key, so it is safe to pass to a player or a third party.

        Args:
            video_id: YouTube video ID (11-12 characters)
            type: "audio" or "video"
            quality: "best", "worst", "128k" (audio), "360p"/"720p"/.. (video)

        Returns:
            StreamUrl with ``url`` and ``expires_in``.
        """
        data = await self._get_json("stream-url", {
            "video_id": self._validate_id(video_id),
            "type": type,
            "quality": quality,
        })
        return StreamUrl(
            url=data.get("url", ""),
            expires_in=data.get("expires_in", 300),
        )

    def stream(
        self,
        video_id: str,
        type: str = "audio",
        quality: str = "best",
    ) -> str:
        """
        Get the stream URL for a YouTube video (no HTTP request).

        The returned URL contains your API key — use :meth:`stream_url` when
        handing the URL to someone else.

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

        The server answers a total failure with **HTTP 200 and an empty body**,
        so an empty result is detected and reported rather than saved.

        Args:
            video_id: YouTube video ID (11-12 characters)
            file_path: Output file path (extension is yours to choose — the
                server sends no Content-Type or filename)
            type: "audio" or "video"
            quality: "best", "worst", "128k" (audio), "720p" (video), etc.
            max_retries: Download retry attempts (default 3)

        Returns:
            Saved file path

        Raises:
            TubeRelayError: If the download produced no data.
        """
        url = self.stream(video_id, type=type, quality=quality)

        result_path = await _download_file(
            url=url,
            output_path=file_path,
            max_retries=max_retries,
            session=self._session,
        )
        if result_path is None:
            raise TubeRelayError(
                f"Failed to download {video_id} to {file_path}: the server "
                f"returned an empty body or an error status",
                endpoint="stream",
            )
        return result_path
