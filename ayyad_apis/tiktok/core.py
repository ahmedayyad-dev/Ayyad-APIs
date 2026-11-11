"""
TikTok Data API wrapper for querying public TikTok data.

This module provides an async interface to the TikTok Data API hosted on RapidAPI
(tiktok-data-api20.p.rapidapi.com). It supports link resolution, link shortening,
video metadata and CDN stream extraction, user profile lookup, user posts listing,
and sound-related video discovery.

Author: Ahmed Ayyad
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

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

_AWEME_ID_PATTERN = re.compile(r"/(?:video|photo)/(\d+)")
_TRACKING_QUERY_PARAMS = frozenset({
    "_t",
    "_r",
    "is_from_webapp",
    "sender_device",
    "share_app_id",
    "share_item_id",
    "share_link_id",
    "social_share_type",
    "source",
    "tt_from",
    "u_code",
    "user_id",
    "sec_user_id",
})


def clean_tiktok_url(url: str) -> str:
    """
    Strip tracking, referral, and analytics query parameters from a TikTok URL.

    Args:
        url: Raw TikTok URL (e.g. from mobile share sheet or web browser).

    Returns:
        Cleaned URL string without unnecessary tracking tags.
    """
    if not url or not url.strip():
        return ""
    parts = urlsplit(url.strip())
    if not parts.query:
        return url.strip()

    kept_params = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_QUERY_PARAMS
    ]
    new_query = urlencode(kept_params) if kept_params else ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def extract_aweme_id(url: str) -> Optional[str]:
    """
    Extract the numeric video ID (aweme_id) from a standard TikTok video URL.

    Does not make network requests; matches URLs of the form
    ``tiktok.com/@user/video/<id>`` or ``/photo/<id>``. Returns None if the URL
    is a short link (use ``resolve_link`` for short links).
    """
    if not url:
        return None
    match = _AWEME_ID_PATTERN.search(url)
    return match.group(1) if match else None


def _format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds as ``m:ss`` or ``h:mm:ss``."""
    if seconds is None:
        return ""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ==================== Data Models ====================


@dataclass
class TikTokResolvedLink(BaseResponse):
    """Result of resolving a shortened or vanity TikTok link."""

    landing_url: str = ""
    status_code: int = 0
    status_msg: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def aweme_id(self) -> Optional[str]:
        """Extract the numeric video ID (aweme_id) from the resolved landing URL if present."""
        if not self.landing_url:
            return None
        match = _AWEME_ID_PATTERN.search(self.landing_url)
        return match.group(1) if match else None


@dataclass
class TikTokShareLink(BaseResponse):
    """Result of creating a shortened vm.tiktok.com share link."""

    shorten_url: str = ""
    status_code: int = 0
    status_msg: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TikTokVideoStatus(BaseResponse):
    """Status check result for a single TikTok video ID."""

    aweme_id: str = ""
    pass_status_check: bool = False
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TikTokUser(BaseResponse):
    """TikTok user metadata."""

    uid: Optional[str] = None
    sec_uid: Optional[str] = None
    unique_id: Optional[str] = None  # TikTok username handle
    nickname: Optional[str] = None
    signature: Optional[str] = None  # Bio
    avatar_thumb: Optional[str] = None
    avatar_medium: Optional[str] = None
    avatar_larger: Optional[str] = None
    verified: bool = False
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TikTokUserProfile(TikTokUser):
    """Full public profile for a TikTok user including counts."""

    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    total_favorited: Optional[int] = None  # Total likes received
    aweme_count: Optional[int] = None  # Published videos count


@dataclass
class TikTokStatistics(BaseResponse):
    """Engagement statistics for a TikTok video."""

    aweme_id: Optional[str] = None
    digg_count: int = 0  # Likes
    comment_count: int = 0
    play_count: int = 0  # Views
    share_count: int = 0
    download_count: int = 0
    collect_count: int = 0  # Favorites/Bookmarks
    forward_count: int = 0
    whatsapp_share_count: int = 0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TikTokMusic(BaseResponse):
    """Audio track details associated with a TikTok video."""

    id: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    album: Optional[str] = None
    duration: Optional[int] = None  # Duration in seconds
    play_url: Optional[str] = None
    cover_large: Optional[str] = None
    cover_medium: Optional[str] = None
    cover_thumb: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def duration_string(self) -> str:
        """Duration formatted as ``m:ss`` or ``h:mm:ss``."""
        return _format_duration(self.duration)

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """Download the audio stream to a local MP3 file."""
        if not self.play_url:
            raise InvalidInputError("No audio play_url available for download.")
        return await _download_file(
            self.play_url,
            output_path=output_path,
            default_filename=f"music_{self.id or 'tiktok'}",
            default_ext=".mp3",
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session,
        )


@dataclass
class TikTokVideoInfo(BaseResponse):
    """Video media properties and CDN stream URLs."""

    duration_ms: Optional[int] = None  # Duration in milliseconds
    width: Optional[int] = None
    height: Optional[int] = None
    ratio: Optional[str] = None
    cover_url: Optional[str] = None
    dynamic_cover_url: Optional[str] = None
    origin_cover_url: Optional[str] = None
    play_addr: List[str] = field(default_factory=list)
    download_addr: List[str] = field(default_factory=list)
    download_no_watermark_addr: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration in seconds."""
        if self.duration_ms is not None:
            return self.duration_ms / 1000.0
        return None

    @property
    def duration_string(self) -> str:
        """Duration formatted as ``m:ss`` or ``h:mm:ss``."""
        return _format_duration(self.duration_seconds)

    @property
    def primary_video_url(self) -> Optional[str]:
        """Return the best available direct video stream or download URL."""
        if self.download_no_watermark_addr:
            return self.download_no_watermark_addr[0]
        if self.download_addr:
            return self.download_addr[0]
        if self.play_addr:
            return self.play_addr[0]
        return None

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """Download video to a local MP4 file."""
        url = self.primary_video_url
        if not url:
            raise InvalidInputError("No video stream URL available for download.")
        return await _download_file(
            url,
            output_path=output_path,
            default_filename="tiktok_video",
            default_ext=".mp4",
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session,
        )


@dataclass
class TikTokVideo(BaseResponse):
    """Complete TikTok video object."""

    aweme_id: str = ""
    desc: str = ""  # Caption
    create_time: Optional[int] = None  # Unix timestamp
    author: Optional[TikTokUser] = None
    statistics: Optional[TikTokStatistics] = None
    music: Optional[TikTokMusic] = None
    video: Optional[TikTokVideoInfo] = None
    share_url: Optional[str] = None
    region: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Video duration in seconds."""
        if self.video:
            return self.video.duration_seconds
        return None

    @property
    def duration_string(self) -> str:
        """Video duration formatted as ``m:ss`` or ``h:mm:ss``."""
        return _format_duration(self.duration_seconds)

    async def download_video(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """Download the video to a local file."""
        if not self.video:
            raise InvalidInputError("No video info present on this item.")
        target = output_path or f"{self.aweme_id}.mp4"
        return await self.video.download(
            output_path=target,
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session,
        )

    async def download_music(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """Download the background audio to a local MP3 file."""
        if not self.music:
            raise InvalidInputError("No music info present on this item.")
        target = output_path or f"music_{self.aweme_id}.mp3"
        return await self.music.download(
            output_path=target,
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session,
        )


@dataclass
class TikTokPostsResponse(BaseResponse):
    """Paginated list of user videos."""

    identifier: str = ""
    aweme_list: List[TikTokVideo] = field(default_factory=list)
    has_more: bool = False
    max_cursor: int = 0
    min_cursor: int = 0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __iter__(self) -> Iterator[TikTokVideo]:
        return iter(self.aweme_list)

    def __len__(self) -> int:
        return len(self.aweme_list)

    def __getitem__(self, index: int) -> TikTokVideo:
        return self.aweme_list[index]


@dataclass
class TikTokMusicVideosResponse(BaseResponse):
    """Paginated videos associated with a specific sound/music ID."""

    music_id: str = ""
    aweme_list: List[TikTokVideo] = field(default_factory=list)
    has_more: bool = False
    cursor: int = 0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __iter__(self) -> Iterator[TikTokVideo]:
        return iter(self.aweme_list)

    def __len__(self) -> int:
        return len(self.aweme_list)

    def __getitem__(self, index: int) -> TikTokVideo:
        return self.aweme_list[index]


@dataclass
class TikTokRelatedMusicResponse(BaseResponse):
    """Related sounds response."""

    music_id: str = ""
    music_list: List[TikTokMusic] = field(default_factory=list)
    has_more: bool = False
    cursor: int = 0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __iter__(self) -> Iterator[TikTokMusic]:
        return iter(self.music_list)

    def __len__(self) -> int:
        return len(self.music_list)

    def __getitem__(self, index: int) -> TikTokMusic:
        return self.music_list[index]


@dataclass
class TikTokUserAnalytics(BaseResponse):
    """Aggregate engagement analytics calculated over a creator's posts."""

    identifier: str = ""
    total_sampled_videos: int = 0
    total_views: int = 0
    avg_views: float = 0.0
    total_likes: int = 0
    avg_likes: float = 0.0
    total_comments: int = 0
    avg_comments: float = 0.0
    total_shares: int = 0
    avg_shares: float = 0.0
    engagement_rate: float = 0.0  # ((likes + comments + shares) / views) * 100
    top_video: Optional[TikTokVideo] = None
    most_liked_video: Optional[TikTokVideo] = None
    most_commented_video: Optional[TikTokVideo] = None


@dataclass
class TikTokBatchDownloadResult(BaseResponse):
    """Result of bulk downloading multiple TikTok videos or audios."""

    total_requested: int = 0
    downloaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)  # (target, error)

    @property
    def success_count(self) -> int:
        return len(self.downloaded)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failed)


# ==================== TikTok API Client ====================


class TikTokAPI(BaseRapidAPI):
    """
    Async client for TikTok Data API hosted on RapidAPI.

    Provides read-only access to public TikTok data without login or cookies:
    link resolution, link shortening, video details with direct CDN streams,
    video status checking, user profiles, user posts, and music discovery.

    Example:
        async with TikTokAPI(api_key="your_rapidapi_key") as api:
            # Resolve short link
            resolved = await api.resolve_link("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/")
            print(resolved.landing_url, resolved.aweme_id)

            # Get video details & download
            video = await api.get_video("7705550414279675143")
            print(video.desc, video.statistics.digg_count)
            await video.download_video("video.mp4")

            # Get author profile
            user = await api.get_user(video.author.sec_uid)
            print(user.nickname, user.follower_count)
    """

    BASE_URL = "https://tiktok-data-api20.p.rapidapi.com"
    DEFAULT_HOST = "tiktok-data-api20.p.rapidapi.com"

    # -------------------- Parsing Helpers --------------------

    @staticmethod
    def _parse_user(data: Optional[Dict[str, Any]]) -> Optional[TikTokUser]:
        if not isinstance(data, dict) or not data:
            return None
        return TikTokUser(
            uid=str(data.get("uid") or data.get("id") or "") or None,
            sec_uid=data.get("sec_uid"),
            unique_id=data.get("unique_id"),
            nickname=data.get("nickname"),
            signature=data.get("signature") or data.get("desc"),
            avatar_thumb=(data.get("avatar_thumb") or {}).get("url_list", [None])[0]
            if isinstance(data.get("avatar_thumb"), dict)
            else data.get("avatar_thumb"),
            avatar_medium=(data.get("avatar_medium") or {}).get("url_list", [None])[0]
            if isinstance(data.get("avatar_medium"), dict)
            else data.get("avatar_medium"),
            avatar_larger=(data.get("avatar_larger") or {}).get("url_list", [None])[0]
            if isinstance(data.get("avatar_larger"), dict)
            else data.get("avatar_larger"),
            verified=bool(data.get("custom_verify") or data.get("enterprise_verify_reason")),
            raw=data,
        )

    @classmethod
    def _parse_user_profile(cls, data: Dict[str, Any]) -> TikTokUserProfile:
        user_dict = data.get("user") if isinstance(data.get("user"), dict) else data
        base_user = cls._parse_user(user_dict) or TikTokUser()
        return TikTokUserProfile(
            uid=base_user.uid,
            sec_uid=base_user.sec_uid,
            unique_id=base_user.unique_id,
            nickname=base_user.nickname,
            signature=base_user.signature,
            avatar_thumb=base_user.avatar_thumb,
            avatar_medium=base_user.avatar_medium,
            avatar_larger=base_user.avatar_larger,
            verified=base_user.verified,
            follower_count=user_dict.get("follower_count"),
            following_count=user_dict.get("following_count"),
            total_favorited=user_dict.get("total_favorited"),
            aweme_count=user_dict.get("aweme_count"),
            raw=data,
        )

    @staticmethod
    def _parse_statistics(data: Optional[Dict[str, Any]]) -> Optional[TikTokStatistics]:
        if not isinstance(data, dict):
            return None
        return TikTokStatistics(
            aweme_id=str(data.get("aweme_id") or "") or None,
            digg_count=int(data.get("digg_count") or 0),
            comment_count=int(data.get("comment_count") or 0),
            play_count=int(data.get("play_count") or 0),
            share_count=int(data.get("share_count") or 0),
            download_count=int(data.get("download_count") or 0),
            collect_count=int(data.get("collect_count") or 0),
            forward_count=int(data.get("forward_count") or 0),
            whatsapp_share_count=int(data.get("whatsapp_share_count") or 0),
            raw=data,
        )

    @staticmethod
    def _parse_music(data: Optional[Dict[str, Any]]) -> Optional[TikTokMusic]:
        if not isinstance(data, dict) or not data:
            return None

        play_url = None
        play_obj = data.get("play_url")
        if isinstance(play_obj, dict):
            urls = play_obj.get("url_list") or []
            if urls:
                play_url = urls[0]
        elif isinstance(play_obj, str):
            play_url = play_obj

        def _get_cover(key: str) -> Optional[str]:
            obj = data.get(key)
            if isinstance(obj, dict):
                urls = obj.get("url_list") or []
                return urls[0] if urls else None
            return obj if isinstance(obj, str) else None

        return TikTokMusic(
            id=str(data.get("id_str") or data.get("id") or "") or None,
            title=data.get("title"),
            author=data.get("author"),
            album=data.get("album"),
            duration=data.get("duration"),
            play_url=play_url,
            cover_large=_get_cover("cover_large"),
            cover_medium=_get_cover("cover_medium"),
            cover_thumb=_get_cover("cover_thumb"),
            raw=data,
        )

    @staticmethod
    def _parse_video_info(data: Optional[Dict[str, Any]]) -> Optional[TikTokVideoInfo]:
        if not isinstance(data, dict) or not data:
            return None

        def _get_url_list(key: str) -> List[str]:
            obj = data.get(key)
            if isinstance(obj, dict):
                return [u for u in (obj.get("url_list") or []) if isinstance(u, str)]
            if isinstance(obj, list):
                return [u for u in obj if isinstance(u, str)]
            if isinstance(obj, str):
                return [obj]
            return []

        def _first_url(key: str) -> Optional[str]:
            urls = _get_url_list(key)
            return urls[0] if urls else None

        return TikTokVideoInfo(
            duration_ms=data.get("duration"),
            width=data.get("width"),
            height=data.get("height"),
            ratio=data.get("ratio"),
            cover_url=_first_url("cover"),
            dynamic_cover_url=_first_url("dynamic_cover") or _first_url("ai_dynamic_cover"),
            origin_cover_url=_first_url("origin_cover"),
            play_addr=_get_url_list("play_addr"),
            download_addr=_get_url_list("download_addr"),
            download_no_watermark_addr=_get_url_list("download_no_watermark_addr"),
            raw=data,
        )

    @classmethod
    def _parse_video(cls, data: Dict[str, Any]) -> TikTokVideo:
        item = data.get("aweme_detail") if isinstance(data.get("aweme_detail"), dict) else data
        return TikTokVideo(
            aweme_id=str(item.get("aweme_id") or ""),
            desc=item.get("desc") or "",
            create_time=item.get("create_time"),
            author=cls._parse_user(item.get("author")),
            statistics=cls._parse_statistics(item.get("statistics")),
            music=cls._parse_music(item.get("music")),
            video=cls._parse_video_info(item.get("video")),
            share_url=item.get("share_url"),
            region=item.get("region"),
            raw=data,
        )

    # -------------------- API Endpoints --------------------

    @with_retry(max_attempts=3, delay=1.0)
    async def resolve_link(self, url: str) -> TikTokResolvedLink:
        """
        Expand a shared TikTok link into its landing target URL.

        Args:
            url: Short shared URL (e.g. ``https://vm.tiktok.com/...``) or any TikTok URL.

        Returns:
            TikTokResolvedLink containing landing_url and the extracted aweme_id property.

        Raises:
            InvalidInputError: If url is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        if not url or not url.strip():
            raise InvalidInputError("TikTok link URL cannot be empty.")

        data = await self._make_request("GET", "/resolve-link", params={"url": url.strip()})
        if not isinstance(data, dict):
            raise RequestError("Unexpected response shape from /resolve-link", endpoint="/resolve-link")

        return TikTokResolvedLink(
            landing_url=data.get("landing_url") or "",
            status_code=int(data.get("status_code") or 0),
            status_msg=data.get("status_msg") or "",
            raw=data,
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def share_link(self, url: str) -> TikTokShareLink:
        """
        Shorten any tiktok.com URL into a vm.tiktok.com share link.

        Args:
            url: Full tiktok.com URL to shorten.

        Returns:
            TikTokShareLink with shorten_url.

        Raises:
            InvalidInputError: If url is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        if not url or not url.strip():
            raise InvalidInputError("URL cannot be empty.")

        data = await self._make_request("GET", "/share-link", params={"url": url.strip()})
        if not isinstance(data, dict):
            raise RequestError("Unexpected response shape from /share-link", endpoint="/share-link")

        return TikTokShareLink(
            shorten_url=data.get("shorten_url") or "",
            status_code=int(data.get("status_code") or 0),
            status_msg=data.get("status_msg") or "",
            raw=data,
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def video_status(
        self,
        ids: Union[str, int, List[Union[str, int]]],
    ) -> List[TikTokVideoStatus]:
        """
        Check if one or more videos are still available.

        Args:
            ids: Single video ID or list of IDs.

        Returns:
            List of TikTokVideoStatus items with aweme_id and pass_status_check.

        Raises:
            InvalidInputError: If ids parameter is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        if isinstance(ids, (list, tuple, set)):
            ids_str = ",".join(str(i).strip() for i in ids if str(i).strip())
        else:
            ids_str = str(ids).strip()

        if not ids_str:
            raise InvalidInputError("Video IDs cannot be empty.")

        data = await self._make_request("GET", "/videos/status", params={"ids": ids_str})
        if not isinstance(data, dict):
            raise RequestError("Unexpected response shape from /videos/status", endpoint="/videos/status")

        results = []
        for item in data.get("aweme_result") or []:
            if isinstance(item, dict):
                results.append(
                    TikTokVideoStatus(
                        aweme_id=str(item.get("aweme_id") or ""),
                        pass_status_check=bool(item.get("pass_status_check")),
                        raw=item,
                    )
                )
        return results

    @with_retry(max_attempts=3, delay=1.0)
    async def get_video(self, aweme_id: Union[str, int]) -> TikTokVideo:
        """
        Fetch complete metadata, statistics, and CDN URLs for a TikTok video.

        Args:
            aweme_id: Numeric TikTok video ID (e.g. 7705550414279675143).

        Returns:
            TikTokVideo instance with description, author, stats, and download helpers.

        Raises:
            InvalidInputError: If aweme_id is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        id_str = str(aweme_id).strip()
        if not id_str:
            raise InvalidInputError("Video ID (aweme_id) cannot be empty.")

        data = await self._make_request("GET", f"/videos/{quote(id_str, safe='')}")
        if not isinstance(data, dict):
            raise RequestError(f"Unexpected response shape from /videos/{id_str}", endpoint=f"/videos/{id_str}")

        return self._parse_video(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def get_user(self, identifier: Union[str, int]) -> TikTokUserProfile:
        """
        Fetch public profile for a TikTok user.

        Args:
            identifier: Either a sec_uid (starts with ``MS4wLjAB...``) or a numeric user ID.

        Returns:
            TikTokUserProfile with follower counts, bio, handle, and avatar URLs.

        Raises:
            InvalidInputError: If identifier is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        ident_str = str(identifier).strip()
        if not ident_str:
            raise InvalidInputError("User identifier cannot be empty.")

        data = await self._make_request("GET", f"/users/{quote(ident_str, safe='')}")
        if not isinstance(data, dict):
            raise RequestError(f"Unexpected response shape from /users/{ident_str}", endpoint=f"/users/{ident_str}")

        return self._parse_user_profile(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def get_user_posts(
        self,
        identifier: Union[str, int],
        count: int = 20,
        cursor: int = 0,
    ) -> TikTokPostsResponse:
        """
        Get public posts (videos) published by a user.

        Args:
            identifier: Either a sec_uid (``MS4wLjAB...``) or numeric user ID.
            count: Number of items per page (1 to 50, default: 20).
            cursor: Paging cursor (pass previous response's ``max_cursor``; 0 = first page).

        Returns:
            TikTokPostsResponse which is iterable and indexable.

        Raises:
            InvalidInputError: If identifier is empty or count is invalid.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        ident_str = str(identifier).strip()
        if not ident_str:
            raise InvalidInputError("User identifier cannot be empty.")

        clamped_count = max(1, min(50, int(count)))
        params = {"count": clamped_count, "cursor": int(cursor)}

        data = await self._make_request("GET", f"/users/{quote(ident_str, safe='')}/posts", params=params)
        if not isinstance(data, dict):
            raise RequestError(f"Unexpected response shape from /users/{ident_str}/posts", endpoint=f"/users/{ident_str}/posts")

        aweme_list = [
            self._parse_video(item)
            for item in (data.get("aweme_list") or [])
            if isinstance(item, dict)
        ]

        return TikTokPostsResponse(
            identifier=ident_str,
            aweme_list=aweme_list,
            has_more=bool(data.get("has_more")),
            max_cursor=int(data.get("max_cursor") or 0),
            min_cursor=int(data.get("min_cursor") or 0),
            raw=data,
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def get_music_videos(
        self,
        music_id: Union[str, int],
        count: int = 20,
        cursor: int = 0,
    ) -> TikTokMusicVideosResponse:
        """
        Get videos that use a specific sound/music track.

        Args:
            music_id: Sound ID.
            count: Number of items per page (1 to 50, default: 20).
            cursor: Paging cursor (default: 0).

        Returns:
            TikTokMusicVideosResponse with aweme_list.

        Raises:
            InvalidInputError: If music_id is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        music_str = str(music_id).strip()
        if not music_str:
            raise InvalidInputError("Music ID cannot be empty.")

        clamped_count = max(1, min(50, int(count)))
        params = {"count": clamped_count, "cursor": int(cursor)}

        data = await self._make_request("GET", f"/music/{quote(music_str, safe='')}/videos", params=params)
        if not isinstance(data, dict):
            raise RequestError(f"Unexpected response shape from /music/{music_str}/videos", endpoint=f"/music/{music_str}/videos")

        aweme_list = [
            self._parse_video(item)
            for item in (data.get("aweme_list") or [])
            if isinstance(item, dict)
        ]

        return TikTokMusicVideosResponse(
            music_id=music_str,
            aweme_list=aweme_list,
            has_more=bool(data.get("has_more")),
            cursor=int(data.get("cursor") or 0),
            raw=data,
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def get_related_music(
        self,
        music_id: Union[str, int],
    ) -> TikTokRelatedMusicResponse:
        """
        Get sounds related to a specific music track.

        Args:
            music_id: Sound ID.

        Returns:
            TikTokRelatedMusicResponse with music_list.

        Raises:
            InvalidInputError: If music_id is empty.
            AuthenticationError: If RapidAPI key is rejected.
            ClientError: On 4xx responses.
            RequestError: On network or server errors.
        """
        music_str = str(music_id).strip()
        if not music_str:
            raise InvalidInputError("Music ID cannot be empty.")

        data = await self._make_request("GET", f"/music/{quote(music_str, safe='')}/related")
        if not isinstance(data, dict):
            raise RequestError(f"Unexpected response shape from /music/{music_str}/related", endpoint=f"/music/{music_str}/related")

        music_list = [
            self._parse_music(item)
            for item in (data.get("music_list") or [])
            if isinstance(item, dict)
        ]
        # Filter out None values
        valid_music_list = [m for m in music_list if m is not None]

        return TikTokRelatedMusicResponse(
            music_id=music_str,
            music_list=valid_music_list,
            has_more=bool(data.get("has_more")),
            cursor=int(data.get("cursor") or 0),
            raw=data,
        )

    # -------------------- Direct Download Shortcuts --------------------

    async def download_video(
        self,
        target: Union[str, int, TikTokVideo],
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Convenience shortcut to download a TikTok video directly.

        Accepts a shared URL (which is resolved automatically), a direct video ID,
        or an existing TikTokVideo object.

        Args:
            target: TikTok video URL (e.g. vm.tiktok.com or tiktok.com), aweme_id, or TikTokVideo.
            output_path: Local file destination path (default: ``<aweme_id>.mp4``).
            show_progress: Whether to log download progress.
            progress_callback: Optional callback receiving (bytes_downloaded, total_bytes).
            session: Optional aiohttp ClientSession.

        Returns:
            String path to the saved MP4 file.
        """
        if isinstance(target, TikTokVideo):
            video = target
        else:
            target_str = str(target).strip()
            if target_str.startswith("http://") or target_str.startswith("https://"):
                resolved = await self.resolve_link(target_str)
                aweme_id = resolved.aweme_id
                if not aweme_id:
                    raise InvalidInputError(f"Could not extract video ID from URL: {target_str}")
            else:
                aweme_id = target_str
            video = await self.get_video(aweme_id)

        return await video.download_video(
            output_path=output_path,
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session or self._session,
        )

    async def download_music(
        self,
        target: Union[str, int, TikTokVideo, TikTokMusic],
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Convenience shortcut to download TikTok background audio directly.

        Accepts a video URL, video ID, TikTokVideo, or TikTokMusic object.

        Args:
            target: TikTok video URL, aweme_id, TikTokVideo, or TikTokMusic.
            output_path: Local file destination path (default: ``music_<id>.mp3``).
            show_progress: Whether to log download progress.
            progress_callback: Optional callback receiving (bytes_downloaded, total_bytes).
            session: Optional aiohttp ClientSession.

        Returns:
            String path to the saved MP3 file.
        """
        if isinstance(target, TikTokMusic):
            return await target.download(
                output_path=output_path,
                show_progress=show_progress,
                progress_callback=progress_callback,
                session=session or self._session,
            )
        if isinstance(target, TikTokVideo):
            return await target.download_music(
                output_path=output_path,
                show_progress=show_progress,
                progress_callback=progress_callback,
                session=session or self._session,
            )

        target_str = str(target).strip()
        if target_str.startswith("http://") or target_str.startswith("https://"):
            resolved = await self.resolve_link(target_str)
            aweme_id = resolved.aweme_id
            if not aweme_id:
                raise InvalidInputError(f"Could not extract video ID from URL: {target_str}")
        else:
            aweme_id = target_str

        video = await self.get_video(aweme_id)
        return await video.download_music(
            output_path=output_path,
            show_progress=show_progress,
            progress_callback=progress_callback,
            session=session or self._session,
        )

    # -------------------- Async Streaming Generators --------------------

    async def iter_user_posts(
        self,
        identifier: Union[str, int],
        max_posts: Optional[int] = None,
        batch_size: int = 20,
    ) -> AsyncIterator[TikTokVideo]:
        """
        Stream public posts published by a user across all pages.

        Automatically tracks the paging cursor (max_cursor) until max_posts is
        reached or the user has no more published posts.

        Args:
            identifier: Either a sec_uid (``MS4wLjAB...``) or numeric user ID.
            max_posts: Optional limit on total videos to yield (None = all available).
            batch_size: Items requested per page (1 to 50, default 20).

        Yields:
            TikTokVideo instances in reverse chronological order.
        """
        cursor = 0
        yielded = 0
        clamped_batch = max(1, min(50, batch_size))

        while True:
            response = await self.get_user_posts(identifier, count=clamped_batch, cursor=cursor)
            if not response.aweme_list:
                break

            for video in response.aweme_list:
                yield video
                yielded += 1
                if max_posts is not None and yielded >= max_posts:
                    return

            if not response.has_more or response.max_cursor == 0 or response.max_cursor == cursor:
                break
            cursor = response.max_cursor

    async def iter_music_videos(
        self,
        music_id: Union[str, int],
        max_videos: Optional[int] = None,
        batch_size: int = 20,
    ) -> AsyncIterator[TikTokVideo]:
        """
        Stream videos associated with a specific sound/music track across pages.

        Args:
            music_id: Sound ID.
            max_videos: Optional limit on total videos to yield.
            batch_size: Items per page.

        Yields:
            TikTokVideo instances.
        """
        cursor = 0
        yielded = 0
        clamped_batch = max(1, min(50, batch_size))

        while True:
            response = await self.get_music_videos(music_id, count=clamped_batch, cursor=cursor)
            if not response.aweme_list:
                break

            for video in response.aweme_list:
                yield video
                yielded += 1
                if max_videos is not None and yielded >= max_videos:
                    return

            if not response.has_more or response.cursor == 0 or response.cursor == cursor:
                break
            cursor = response.cursor

    # -------------------- Analytics Engine --------------------

    async def get_user_analytics(
        self,
        identifier: Union[str, int],
        sample_size: int = 20,
    ) -> TikTokUserAnalytics:
        """
        Calculate creator engagement metrics and performance statistics.

        Samples up to sample_size recent posts to calculate total/average views,
        likes, comments, shares, overall engagement rate, and identify top performing posts.

        Args:
            identifier: User sec_uid or numeric user ID.
            sample_size: Number of recent posts to evaluate (default: 20).

        Returns:
            TikTokUserAnalytics dataclass.
        """
        ident_str = str(identifier).strip()
        videos: List[TikTokVideo] = []
        async for v in self.iter_user_posts(ident_str, max_posts=sample_size):
            videos.append(v)

        if not videos:
            return TikTokUserAnalytics(identifier=ident_str)

        total_views = 0
        total_likes = 0
        total_comments = 0
        total_shares = 0

        top_video: Optional[TikTokVideo] = None
        max_views = -1
        most_liked: Optional[TikTokVideo] = None
        max_likes = -1
        most_commented: Optional[TikTokVideo] = None
        max_comments = -1

        for v in videos:
            st = v.statistics or TikTokStatistics()
            total_views += st.play_count
            total_likes += st.digg_count
            total_comments += st.comment_count
            total_shares += st.share_count

            if st.play_count > max_views:
                max_views = st.play_count
                top_video = v
            if st.digg_count > max_likes:
                max_likes = st.digg_count
                most_liked = v
            if st.comment_count > max_comments:
                max_comments = st.comment_count
                most_commented = v

        n = len(videos)
        engagement_rate = ((total_likes + total_comments + total_shares) / total_views * 100.0) if total_views > 0 else 0.0

        return TikTokUserAnalytics(
            identifier=ident_str,
            total_sampled_videos=n,
            total_views=total_views,
            avg_views=round(total_views / n, 2),
            total_likes=total_likes,
            avg_likes=round(total_likes / n, 2),
            total_comments=total_comments,
            avg_comments=round(total_comments / n, 2),
            total_shares=total_shares,
            avg_shares=round(total_shares / n, 2),
            engagement_rate=round(engagement_rate, 2),
            top_video=top_video,
            most_liked_video=most_liked,
            most_commented_video=most_commented,
        )

    # -------------------- Bulk & Batch Downloaders --------------------

    async def download_user_videos(
        self,
        identifier: Union[str, int],
        output_dir: Union[str, Path] = "downloads",
        max_videos: int = 10,
        skip_existing: bool = True,
        download_music: bool = False,
        concurrency: int = 3,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> TikTokBatchDownloadResult:
        """
        Download multiple videos or audio tracks from a user profile.

        Args:
            identifier: sec_uid or numeric user ID.
            output_dir: Local directory where files will be saved.
            max_videos: Maximum number of recent videos to download (default: 10).
            skip_existing: If True, skips files that already exist on disk.
            download_music: If True, downloads MP3 audio instead of MP4 video.
            concurrency: Number of concurrent download workers (default: 3).
            progress_callback: Optional callback receiving (filename, current_index, total).

        Returns:
            TikTokBatchDownloadResult with lists of downloaded, skipped, and failed files.
        """
        videos: List[TikTokVideo] = []
        async for v in self.iter_user_posts(identifier, max_posts=max_videos):
            videos.append(v)

        return await self.download_batch(
            targets=videos,
            output_dir=output_dir,
            skip_existing=skip_existing,
            download_music=download_music,
            concurrency=concurrency,
            progress_callback=progress_callback,
        )

    async def download_batch(
        self,
        targets: List[Union[str, int, TikTokVideo]],
        output_dir: Union[str, Path] = "downloads",
        skip_existing: bool = True,
        download_music: bool = False,
        concurrency: int = 3,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> TikTokBatchDownloadResult:
        """
        Batch download a list of videos or audio tracks concurrently.

        Args:
            targets: List of video URLs, aweme_ids, or TikTokVideo instances.
            output_dir: Local destination directory.
            skip_existing: Skip already downloaded files.
            download_music: If True, downloads audio (.mp3); otherwise video (.mp4).
            concurrency: Maximum concurrent workers.
            progress_callback: Optional callback receiving (filename, current_index, total).

        Returns:
            TikTokBatchDownloadResult summarizing downloaded, skipped, and failed items.
        """
        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(max(1, concurrency))
        result = TikTokBatchDownloadResult(total_requested=len(targets))
        lock = asyncio.Lock()
        completed = 0
        total = len(targets)

        async def _worker(target: Union[str, int, TikTokVideo]) -> None:
            nonlocal completed
            async with sem:
                try:
                    if isinstance(target, TikTokVideo):
                        aweme_id = target.aweme_id
                        video_obj = target
                    else:
                        target_str = str(target).strip()
                        if target_str.startswith("http://") or target_str.startswith("https://"):
                            resolved = await self.resolve_link(target_str)
                            aweme_id = resolved.aweme_id or "unknown"
                        else:
                            aweme_id = target_str
                        video_obj = target

                    ext = ".mp3" if download_music else ".mp4"
                    prefix = "music_" if download_music else ""
                    filename = f"{prefix}{aweme_id}{ext}"
                    filepath = dest_dir / filename

                    if skip_existing and filepath.exists() and filepath.stat().st_size > 0:
                        async with lock:
                            result.skipped.append(str(filepath))
                            completed += 1
                            if progress_callback:
                                progress_callback(str(filepath), completed, total)
                        return

                    if download_music:
                        saved = await self.download_music(video_obj, output_path=filepath)
                    else:
                        saved = await self.download_video(video_obj, output_path=filepath)

                    async with lock:
                        result.downloaded.append(str(saved))
                        completed += 1
                        if progress_callback:
                            progress_callback(str(saved), completed, total)

                except Exception as e:
                    logger.error(f"Failed to download batch item {target}: {e}")
                    async with lock:
                        result.failed.append((str(target), str(e)))
                        completed += 1
                        if progress_callback:
                            progress_callback(str(target), completed, total)

        await asyncio.gather(*[_worker(t) for t in targets])
        return result

