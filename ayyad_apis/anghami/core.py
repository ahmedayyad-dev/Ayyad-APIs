"""
Anghami API wrapper for catalog search, song metadata, and CDN streams.

This module provides an async interface to the Anghami Music Data and Stream
API hosted on RapidAPI. It supports searching the catalog, fetching song
details, resolving time-limited CDN playback URLs, and downloading tracks.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import quote

from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    DownloadError,
    InvalidInputError,
    RequestError,
    download_file as _download_file,
    with_retry,
)

logger = logging.getLogger(__name__)

# Tags as returned by the Anghami API.
_SONG_SUMMARY_KEYS = frozenset({
    "id",
    "title",
    "artist",
    "artistID",
    "album",
    "albumID",
    "generictype",
    "genre",
    "duration",
    "coverArt",
    "explicit",
})

_SONG_DETAIL_KEYS = _SONG_SUMMARY_KEYS | frozenset({
    "status",
    "responsetype",
    "sections",
})


def _format_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds as ``m:ss`` or ``h:mm:ss``."""
    if seconds is None:
        return ""

    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _cover_url(cover_art: Optional[int], size: int = 480) -> Optional[str]:
    """Build an Anghami artwork URL from a cover-art id."""
    if not cover_art:
        return None
    return f"https://artwork.anghcdn.co/webp/?id={cover_art}&size={size}"


# ==================== Data Models ====================


@dataclass
class SongSummary(BaseResponse):
    """A song returned by the ``/search`` endpoint."""

    id: str = ""
    title: Optional[str] = None
    artist: Optional[str] = None
    artist_id: Optional[str] = None
    album: Optional[str] = None
    album_id: Optional[str] = None
    generic_type: Optional[str] = None
    genre: Optional[str] = None
    duration: Optional[float] = None
    cover_art: Optional[int] = None
    explicit: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def duration_string(self) -> str:
        """Duration formatted as ``m:ss`` or ``h:mm:ss``."""
        return _format_duration(self.duration)

    @property
    def is_explicit(self) -> bool:
        """Whether the track is flagged as explicit."""
        return bool(self.explicit)

    def cover_art_url(self, size: int = 480) -> Optional[str]:
        """Return the artwork URL for this song, or ``None`` when unavailable."""
        return _cover_url(self.cover_art, size)


@dataclass
class SongDetail(SongSummary):
    """Full song metadata returned by ``/songs/{song_id}``."""

    status: Optional[str] = None
    response_type: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SearchResponse(BaseResponse):
    """Response from the ``/search`` endpoint."""

    query: str = ""
    count: int = 0
    results: List[SongSummary] = field(default_factory=list)

    def __iter__(self) -> Iterator[SongSummary]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> SongSummary:
        return self.results[index]


@dataclass
class StreamUrl(BaseResponse):
    """A time-limited CDN playback URL resolved by ``/songs/{id}/url``."""

    song_id: str = ""
    quality: str = "64"
    url: str = ""

    def __str__(self) -> str:
        return self.url


# ==================== API Client ====================


class AnghamiAPI(BaseRapidAPI):
    """
    Client for the Anghami Music Data and Stream API on RapidAPI.

    All requests are authenticated with a RapidAPI key and are sent through the
    RapidAPI gateway. The API exposes three endpoints:

    - ``GET /search`` — search the catalog.
    - ``GET /songs/{song_id}`` — fetch full song metadata.
    - ``GET /songs/{song_id}/url`` — resolve a time-limited CDN stream URL.

    Example::

        async with AnghamiAPI(api_key="your-key") as api:
            found = await api.search("Amr Diab")
            for song in found:
                print(song.title, song.duration_string)

            detail = await api.get_song(found[0].id)
            print(detail.genre)

            stream = await api.stream_url(detail.id, quality="128")
            print(stream.url)
    """

    BASE_URL = "https://anghami-music-data-and-stream.p.rapidapi.com"
    DEFAULT_HOST = "anghami-music-data-and-stream.p.rapidapi.com"
    QUALITIES = ("24", "64", "128", "196")

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise InvalidInputError("query must be a non-empty string")
        return query.strip()

    @staticmethod
    def _validate_song_id(song_id: Union[str, int]) -> str:
        if song_id is None or isinstance(song_id, bool):
            raise InvalidInputError("song_id is required")

        value = str(song_id).strip()
        if not value:
            raise InvalidInputError("song_id is required")

        if "://" in value or "/" in value or "?" in value or "#" in value:
            raise InvalidInputError(
                f"song_id must be an Anghami song ID, not a URL: {song_id!r}"
            )

        return value

    @classmethod
    def _validate_quality(cls, quality: Union[str, int]) -> str:
        value = str(quality).strip()
        if value not in cls.QUALITIES:
            allowed = ", ".join(cls.QUALITIES)
            raise InvalidInputError(
                f"quality must be one of {allowed}, got {quality!r}"
            )
        return value

    @staticmethod
    def _song_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(data.get("id", "")),
            "title": data.get("title"),
            "artist": data.get("artist"),
            "artist_id": data.get("artistID"),
            "album": data.get("album"),
            "album_id": data.get("albumID"),
            "generic_type": data.get("generictype"),
            "genre": data.get("genre"),
            "duration": data.get("duration"),
            "cover_art": data.get("coverArt"),
            "explicit": data.get("explicit"),
        }

    @classmethod
    def _parse_summary(cls, data: Dict[str, Any]) -> SongSummary:
        raw = {key: value for key, value in data.items() if key not in _SONG_SUMMARY_KEYS}
        return SongSummary(**cls._song_fields(data), raw=raw)

    @classmethod
    def _parse_detail(cls, data: Dict[str, Any]) -> SongDetail:
        raw = {key: value for key, value in data.items() if key not in _SONG_DETAIL_KEYS}
        sections = data.get("sections") or []
        return SongDetail(
            **cls._song_fields(data),
            raw=raw,
            status=data.get("status"),
            response_type=data.get("responsetype"),
            sections=sections if isinstance(sections, list) else [],
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, filter_type: str = "top") -> SearchResponse:
        """
        Search the Anghami catalog for songs.

        Args:
            query: Search text (song title, artist name, etc.).
            filter_type: Anghami filter type passed through to the server
                (defaults to ``"top"``).

        Returns:
            SearchResponse with ``query``, ``count``, and a list of songs.
            The response is iterable and indexable, so it can be used directly
            in a ``for`` loop.

        Raises:
            InvalidInputError: If the query is empty.
            AuthenticationError: If the RapidAPI key is rejected.
            ClientError: On other 4xx responses.
            RequestError: On 5xx or network failures.
        """
        normalized = self._validate_query(query)
        data = await self._make_request(
            "GET",
            "/search",
            params={"q": normalized, "filter_type": filter_type},
        )

        if not isinstance(data, dict):
            raise RequestError(
                "Unexpected response shape from /search",
                endpoint="/search",
            )

        results = [
            self._parse_summary(item)
            for item in (data.get("results") or [])
            if isinstance(item, dict)
        ]
        return SearchResponse(
            query=data.get("query", normalized),
            count=data.get("count", len(results)),
            results=results,
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def get_song(self, song_id: Union[str, int]) -> SongDetail:
        """
        Fetch full metadata for a song.

        Args:
            song_id: Anghami song ID.

        Returns:
            SongDetail including title, artist, album, genre, duration, cover
            art, and raw recommendation sections.

        Raises:
            InvalidInputError: If the song ID is empty or looks like a URL.
            AuthenticationError: If the RapidAPI key is rejected.
            ClientError: On other 4xx responses.
            RequestError: On 5xx or network failures.
        """
        normalized = self._validate_song_id(song_id)
        data = await self._make_request("GET", f"/songs/{quote(normalized, safe='')}")

        if not isinstance(data, dict):
            raise RequestError(
                f"Unexpected response shape from /songs/{normalized}",
                endpoint=f"/songs/{normalized}",
            )

        return self._parse_detail(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def stream_url(
        self,
        song_id: Union[str, int],
        quality: Union[str, int] = "64",
    ) -> StreamUrl:
        """
        Resolve a time-limited, playable CDN URL for a song.

        The URL expires, so resolve it immediately before playing or
        downloading. It does not contain the RapidAPI key.

        Args:
            song_id: Anghami song ID.
            quality: Audio bitrate, one of ``"24"``, ``"64"``, ``"128"``,
                ``"196"`` (default ``"64"``).

        Returns:
            StreamUrl with ``song_id``, ``quality``, and ``url``.

        Raises:
            InvalidInputError: If the song ID or quality is invalid.
            AuthenticationError: If the RapidAPI key is rejected.
            ClientError: On other 4xx responses.
            RequestError: On 5xx, network failures, or a response without a URL.
        """
        normalized_id = self._validate_song_id(song_id)
        normalized_quality = self._validate_quality(quality)
        endpoint = f"/songs/{normalized_id}/url"

        data = await self._make_request(
            "GET",
            f"/songs/{quote(normalized_id, safe='')}/url",
            params={"quality": normalized_quality},
        )

        if not isinstance(data, dict) or not data.get("url"):
            raise RequestError(
                "The API did not return a stream URL",
                endpoint=endpoint,
            )

        return StreamUrl(
            song_id=str(data.get("song_id", normalized_id)),
            quality=str(data.get("quality", normalized_quality)),
            url=data["url"],
        )

    async def download(
        self,
        song_id: Union[str, int],
        file_path: str,
        quality: Union[str, int] = "64",
        max_retries: int = 3,
    ) -> str:
        """
        Download a song to a local file.

        Args:
            song_id: Anghami song ID.
            file_path: Output file path. Use an ``.m4a`` extension; the CDN
                serves AAC audio with a ``video/mp4`` content type.
            quality: Audio bitrate, one of ``"24"``, ``"64"``, ``"128"``,
                ``"196"`` (default ``"64"``).
            max_retries: Download retry attempts (default 3).

        Returns:
            The saved file path.

        Raises:
            InvalidInputError: If the song ID or quality is invalid.
            DownloadError: If the download produced no data.
        """
        stream = await self.stream_url(song_id, quality=quality)
        result_path = await _download_file(
            url=stream.url,
            output_path=file_path,
            max_retries=max_retries,
            session=self._session,
        )
        if result_path is None:
            raise DownloadError(
                f"Failed to download song {song_id} to {file_path}"
            )
        return result_path

    async def quick_download(
        self,
        query: str,
        output_path: Optional[Union[str, Path]] = None,
        quality: Union[str, int] = "128",
        max_retries: int = 3,
    ) -> str:
        """
        One-liner search and download for Anghami.

        Searches for the query, picks the top matching track, resolves its CDN
        playback URL, and downloads it directly to disk.

        Args:
            query: Song title, artist name, or search keywords.
            output_path: Optional destination file path (defaults to ``<artist> - <title>.m4a``).
            quality: Audio bitrate (``"24"``, ``"64"``, ``"128"``, or ``"196"``, default: ``"128"``).
            max_retries: Maximum download retry attempts.

        Returns:
            String path to the saved audio file.

        Raises:
            InvalidInputError: If no songs matched the query or query is empty.
        """
        search_res = await self.search(query)
        if not search_res.results:
            raise InvalidInputError(f"No songs found on Anghami matching: {query!r}")

        top_song = search_res.results[0]
        if output_path is None:
            artist = (top_song.artist or "Unknown").replace("/", "-")
            title = (top_song.title or top_song.id).replace("/", "-")
            destination = f"{artist} - {title}.m4a"
        else:
            destination = str(output_path)

        return await self.download(top_song.id, file_path=destination, quality=quality, max_retries=max_retries)

    async def get_recommendations(
        self,
        song_id: Union[str, int],
    ) -> List[SongSummary]:
        """
        Get recommended/related songs for a given track.

        Extracts and parses recommended tracks from the track's recommendation
        sections into clean SongSummary objects.

        Args:
            song_id: Anghami song ID.

        Returns:
            List of SongSummary objects representing related tracks.
        """
        detail = await self.get_song(song_id)
        recommendations: List[SongSummary] = []
        seen_ids = {detail.id}

        for sec in detail.sections:
            if isinstance(sec, dict):
                # Sections often contain a 'data' list of songs
                items = sec.get("data") or []
                for item in items:
                    if isinstance(item, dict) and item.get("id"):
                        s_id = str(item["id"])
                        if s_id not in seen_ids:
                            seen_ids.add(s_id)
                            recommendations.append(self._parse_summary(item))

        return recommendations
