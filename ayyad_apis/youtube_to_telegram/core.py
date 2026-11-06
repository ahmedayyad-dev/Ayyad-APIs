from __future__ import annotations

import asyncio
import logging
import json
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Union

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    ClientError,
    RequestError,
    DownloadError,
    APIConfig,
    with_retry,
    validate_rapidapi_response,
    download_file as _download_file,
)

logger = logging.getLogger(__name__)


# ==================== Exception Aliases ====================

class APIResponseError(RequestError):
    """Error when API doesn't return 200 status or returns an error message"""
    def __init__(self, message: str):
        super().__init__(f"API Error: {message}")
        self.message = message


# ==================== Enums ====================

class JobStatus(str, Enum):
    """Status values the API reports for a queued download.

    Note:
        The API answers HTTP 202 with ``status="processing"`` when the same
        video/format/quality is already being handled by another request. There
        is no pollable progress endpoint, so this enum describes that one
        queued state plus the terminal states.
    """
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== Data Models ====================

@dataclass
class Channel(BaseResponse):
    """Channel information"""
    name: Optional[str] = None
    id: Optional[str] = None
    thumbnails: Optional[List[dict]] = None
    link: Optional[str] = None


@dataclass
class Video(BaseResponse):
    """Base video object shared across multiple responses"""
    success: bool = False
    video_title: Optional[str] = None
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
    duration: Optional[int] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    uploader: Optional[Channel] = None

    @property
    def duration_formatted(self) -> str:
        """Format video duration to HH:MM:SS."""
        if not self.duration:
            return "00:00:00"
        hours, remainder = divmod(self.duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def views_formatted(self) -> str:
        """Format view count with K/M suffix."""
        if not self.view_count:
            return "0"
        if self.view_count >= 1_000_000:
            return f"{self.view_count / 1_000_000:.1f}M"
        elif self.view_count >= 1_000:
            return f"{self.view_count / 1_000:.1f}K"
        return str(self.view_count)


@dataclass
class VideoInfoResponse(BaseResponse):
    """Detailed response for /video_info endpoint"""
    success: bool = False
    title: Optional[str] = None
    description: Optional[str] = None
    duration_seconds: Optional[int] = None
    duration_string: Optional[str] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    concurrent_view_count: Optional[int] = None
    thumbnail: Optional[str] = None
    language: Optional[str] = None
    has_subtitles: bool = False
    subtitle_languages: Optional[List[str]] = None
    uploader_info: Optional[dict] = None
    id: Optional[str] = None
    webpage_url: Optional[str] = None
    webpage_url_domain: Optional[str] = None
    formats: Optional[List[dict]] = None


@dataclass
class TelegramResponse(BaseResponse):
    """Response when uploading YouTube video to Telegram"""
    file_url: Optional[str] = None
    message_id: Optional[int] = None
    chat_username: Optional[str] = None
    warning: Optional[str] = None


AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".aac", ".opus", ".ogg", ".wav", ".flac"})


async def strip_thumbnail_from_video(source: Union[str, Path], output_path: Union[str, Path]) -> str:
    """Extract audio track and strip video thumbnail from a video file or remote URL using ffmpeg.

    Directly passes the source (local path or remote HTTP/HTTPS URL) to ffmpeg so no
    intermediate video file needs to be downloaded to disk first.

    Args:
        source: Remote video URL (e.g. Google CDN URL) or local file path.
        output_path: Path to save the extracted audio file.

    Returns:
        Path to the output audio file as a string.

    Raises:
        DownloadError: If ffmpeg is not found in PATH or extraction fails.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise DownloadError("ffmpeg is required to strip video thumbnail but was not found in PATH.")

    source_str = str(source)
    output_str = str(output_path)
    ext = Path(output_str).suffix.lower()

    if ext == ".mp3":
        codec_candidates = [
            ["-c:a", "libmp3lame", "-q:a", "2"],
            ["-c:a", "mp3"],
        ]
    elif ext in (".m4a", ".aac"):
        codec_candidates = [
            ["-c:a", "copy"],
            ["-c:a", "aac", "-b:a", "192k"],
        ]
    elif ext == ".opus":
        codec_candidates = [
            ["-c:a", "libopus", "-b:a", "128k"],
            ["-c:a", "copy"],
        ]
    elif ext == ".wav":
        codec_candidates = [
            ["-c:a", "pcm_s16le"],
        ]
    else:
        codec_candidates = [
            ["-c:a", "copy"],
            [],
        ]

    reconnect_args = [
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
    ] if source_str.startswith(("http://", "https://")) else []

    last_error = ""
    for codec_args in codec_candidates:
        cmd = [
            ffmpeg_bin,
            "-y",
            *reconnect_args,
            "-i", source_str,
            "-vn",
            *codec_args,
            output_str,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0 and Path(output_str).exists() and Path(output_str).stat().st_size > 0:
            return output_str
        last_error = stderr.decode("utf-8", errors="replace")
        try:
            Path(output_str).unlink(missing_ok=True)
        except OSError:
            pass

    raise DownloadError(f"Failed to strip thumbnail with ffmpeg: {last_error}")


@dataclass
class GoogleResponse(BaseResponse):
    """Response when uploading YouTube video or audio to Google Photos"""
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    warning: Optional[str] = None
    _api_instance: Optional[YouTubeAPI] = None

    async def download_file(
        self,
        file_path: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        strip_thumbnail: bool = True,
    ) -> DownloadResult:
        """Download the resulting file to ``file_path``.

        If the file extension is an audio format (e.g. .mp3, .m4a, .aac, .opus, .wav, .flac)
        or if audio_url is present, it delegates to :meth:`download_audio` to ensure any
        1-frame thumbnail video attached by Google Photos is stripped.
        """
        ext = Path(file_path).suffix.lower()
        is_audio = ext in AUDIO_EXTENSIONS

        if is_audio or (self.audio_url and not self.video_url):
            return await self.download_audio(
                file_path,
                max_retries=max_retries,
                retry_delay=retry_delay,
                strip_thumbnail=strip_thumbnail,
            )

        return await self.download_video(file_path, max_retries=max_retries, retry_delay=retry_delay)

    async def download_video(
        self,
        file_path: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> DownloadResult:
        """Download the video file from Google Photos CDN to ``file_path``."""
        if not self._api_instance:
            raise DownloadError("API instance not available")
        if not self.video_url:
            raise DownloadError("No video URL available")
        logger.info(f"Downloading video to {file_path} from Google Photos...")

        retries = max_retries if max_retries is not None else self._api_instance._max_retries
        delay = retry_delay if retry_delay is not None else self._api_instance._retry_delay

        return await self._api_instance.download_file(self.video_url, file_path, retries, delay)

    async def download_audio(
        self,
        file_path: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        strip_thumbnail: bool = True,
    ) -> DownloadResult:
        """Download the audio stream to ``file_path``.

        Audio files uploaded to Google Photos are packaged as 1-frame MP4 videos with the
        video thumbnail. If ``strip_thumbnail=True`` (default):
        - Directly passes the remote video URL to ffmpeg to stream and strip the 1-frame thumbnail
          without downloading a temporary video file to disk first.
        - If ffmpeg extraction fails or ffmpeg is not installed, it automatically falls back
          to downloading the audio directly from the API (``audio_url``).

        Args:
            file_path: Target path to save the audio file (e.g. 'song.mp3', 'song.m4a').
            max_retries: Number of retry attempts.
            retry_delay: Delay between retries in seconds.
            strip_thumbnail: Whether to strip the 1-frame thumbnail video stream (default True).
        """
        if not self._api_instance:
            raise DownloadError("API instance not available")

        retries = max_retries if max_retries is not None else self._api_instance._max_retries
        delay = retry_delay if retry_delay is not None else self._api_instance._retry_delay
        ffmpeg_available = bool(shutil.which("ffmpeg"))

        # Step 1: Pass the remote video URL directly to ffmpeg to stream and strip thumbnail on the fly
        if strip_thumbnail and ffmpeg_available and self.video_url:
            try:
                logger.info(f"Extracting clean audio directly from remote URL to {file_path} via ffmpeg...")
                target_dir = Path(file_path).resolve().parent
                target_dir.mkdir(parents=True, exist_ok=True)
                await strip_thumbnail_from_video(self.video_url, file_path)
                return DownloadResult(file_path=file_path, file_size=Path(file_path).stat().st_size)
            except Exception as e:
                logger.warning(
                    f"Direct audio extraction with ffmpeg failed ({e}); falling back to downloading audio from API directly..."
                )
                try:
                    Path(file_path).unlink(missing_ok=True)
                except OSError:
                    pass

        # Step 2: Fallback to downloading audio directly from the API (audio_url)
        if self.audio_url:
            logger.info(f"Downloading pre-stripped audio stream from API to {file_path}...")
            return await self._api_instance.download_file(self.audio_url, file_path, retries, delay)

        # Step 3: If no audio_url was returned and ffmpeg extraction failed/unavailable, download video_url
        if self.video_url:
            if strip_thumbnail:
                logger.warning(
                    "ffmpeg is unavailable or failed, and no audio_url was provided by the API. "
                    "Downloading raw file from Google Photos (contains 1-frame video thumbnail)."
                )
            logger.info(f"Downloading file to {file_path} from Google Photos...")
            return await self._api_instance.download_file(self.video_url, file_path, retries, delay)

        raise DownloadError("No download URL available")


@dataclass
class DownloadResult(BaseResponse):
    """Represents result of a local file download"""
    file_path: Optional[str] = None
    file_size: Optional[int] = None


@dataclass
class LiveStream(BaseResponse):
    """Represents a live stream (HLS/MP4) or streaming URL"""
    url: Optional[str] = None
    warning: Optional[str] = None


@dataclass
class QueuedJobResponse(BaseResponse):
    """HTTP 202 body — this exact video is already being downloaded.

    Attributes:
        job_id: Server-side identifier of the in-flight job.
        video_id: The video that is already queued.
        status: Always ``"processing"``.
        message: Human-readable explanation from the API.
    """
    job_id: Optional[str] = None
    video_id: Optional[str] = None
    status: str = JobStatus.PROCESSING.value
    message: Optional[str] = None


@dataclass
class ServerResponse(BaseResponse):
    """Response for /youtube_to_server — includes a temporary download URL (valid 10 min)"""
    download_url: Optional[str] = None
    key: Optional[str] = None
    video_title: Optional[str] = None
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
    duration: Optional[int] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    uploader: Optional[Channel] = None
    warning: Optional[str] = None
    _api_instance: Optional[YouTubeAPI] = None

    def __post_init__(self):
        if not self.key and self.download_url and "key=" in self.download_url:
            self.key = self.download_url.split("key=")[1].split("&")[0]

    @property
    def duration_formatted(self) -> str:
        if not self.duration:
            return "00:00:00"
        hours, remainder = divmod(self.duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def views_formatted(self) -> str:
        if not self.view_count:
            return "0"
        if self.view_count >= 1_000_000:
            return f"{self.view_count / 1_000_000:.1f}M"
        elif self.view_count >= 1_000:
            return f"{self.view_count / 1_000:.1f}K"
        return str(self.view_count)

    async def download_file(self, file_path: str, max_retries: Optional[int] = None,
                            retry_delay: Optional[float] = None) -> DownloadResult:
        """Download the server-hosted file to ``file_path``."""
        if not self._api_instance:
            raise DownloadError("API instance not available")
        if not self.download_url:
            raise DownloadError("No download URL available")
        logger.info(f"Downloading file to {file_path} from server...")

        retries = max_retries if max_retries is not None else self._api_instance._max_retries
        delay = retry_delay if retry_delay is not None else self._api_instance._retry_delay

        return await self._api_instance.download_file(self.download_url, file_path, retries, delay)


@dataclass
class VideoSearchResult(BaseResponse):
    """Single video result from /search"""
    title: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None
    duration_seconds: Optional[int] = None
    duration_string: Optional[str] = None
    view_count: Optional[int] = None
    thumbnail: Optional[str] = None
    webpage_url: Optional[str] = None
    webpage_url_domain: Optional[str] = None
    uploader_info: Optional[dict] = None


class DownloadInProgressError(RequestError):
    """Raised on HTTP 202 — the same video is already being downloaded.

    The API deduplicates concurrent requests for the same
    video/format/quality and answers HTTP 202 instead of doing the work twice.
    There is no progress endpoint to poll, so the in-flight job finishes on its
    own; retry the original request after a short delay.

    Access the structured body via ``exception.response``.

    Example::

        try:
            result = await api.youtube_to_telegram("dQw4w9WgXcQ")
        except DownloadInProgressError as e:
            print(f"Already running as job {e.response.job_id}")
    """

    # Retrying cannot help: the API has already accepted this video and is
    # working on it. A retry would just be rejected again, so this is raised
    # immediately instead of being retried like a 5xx.
    non_retryable = True

    def __init__(self, response: QueuedJobResponse):
        self.response = response
        super().__init__(
            f"This video is already being processed (job {response.job_id}): "
            f"{response.message or 'retry shortly'}",
            status_code=202,
            endpoint="youtube_to_telegram",
        )


# ==================== API Client ====================

class YouTubeAPI(BaseRapidAPI):
    """
    API client wrapper for YouTube to Telegram and YouTube to Server endpoints.

    Parameters:
        api_key: RapidAPI key.
        timeout: Request timeout in seconds (default 300).
        max_retries: Retry attempts for downloads (default 5).
        retry_delay: Seconds between retries (default 1.0).
        cookies: Netscape-format cookie string sent as the ``X-Cookies`` header.
        own_session: Use a dedicated session instead of the shared global one.
            Required when the client is used across more than one event loop.

    Example::

        async with YouTubeAPI(api_key="key") as client:
            result = await client.youtube_to_telegram("dQw4w9WgXcQ")
            print(result.file_url)

    Example — handling a deduplicated request::

        async with YouTubeAPI(api_key="key") as client:
            try:
                result = await client.youtube_to_telegram("dQw4w9WgXcQ")
            except DownloadInProgressError as e:
                print(f"Already running as job {e.response.job_id}")
    """

    BASE_URL = "https://youtube-to-telegram-uploader-api.p.rapidapi.com"
    DEFAULT_HOST = "youtube-to-telegram-uploader-api.p.rapidapi.com"

    def __init__(self, api_key: str, timeout: int = 300, max_retries: int = 5, retry_delay: float = 1.0,
                 cookies: Optional[str] = None, config: Optional[APIConfig] = None,
                 own_session: bool = False):
        super().__init__(api_key=api_key, timeout=timeout, config=config, own_session=own_session)

        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cookies = cookies

    def _parse_response_data(self, data: dict, response_class):
        """Helper to convert API response into dataclass objects"""

        parsed_data = data.copy()

        if response_class in (ServerResponse, GoogleResponse):
            parsed_data['_api_instance'] = self

        if 'uploader' in parsed_data and isinstance(parsed_data['uploader'], dict):
            uploader_data = parsed_data['uploader'].copy()
            if 'channel_name' in uploader_data:
                uploader_data['name'] = uploader_data.pop('channel_name')
            if 'channel_url' in uploader_data:
                uploader_data['link'] = uploader_data.pop('channel_url')
            parsed_data['uploader'] = Channel(**uploader_data)

        # Normalize tags to list
        if 'tags' in parsed_data and not isinstance(parsed_data['tags'], list):
            if parsed_data['tags'] is None:
                parsed_data['tags'] = []
            else:
                parsed_data['tags'] = [parsed_data['tags']] if isinstance(parsed_data['tags'], str) else []

        # Normalize subtitle_languages and formats for VideoInfoResponse
        if response_class == VideoInfoResponse:
            if 'subtitle_languages' in parsed_data and not isinstance(parsed_data['subtitle_languages'], list):
                parsed_data['subtitle_languages'] = (
                    [] if parsed_data['subtitle_languages'] is None
                    else [parsed_data['subtitle_languages']] if isinstance(parsed_data['subtitle_languages'], str)
                    else []
                )

            if 'formats' in parsed_data and not isinstance(parsed_data['formats'], list):
                parsed_data['formats'] = (
                    [] if parsed_data['formats'] is None
                    else [parsed_data['formats']] if isinstance(parsed_data['formats'], dict)
                    else []
                )

        # The API returns more fields than the dataclasses declare (and may add
        # more over time), so select the declared fields instead of letting
        # construction fail on the extras. Fields injected above — notably the
        # private _api_instance that ServerResponse.download_file() depends on —
        # are preserved, and anything absent simply keeps its dataclass default.
        declared = response_class.__dataclass_fields__
        safe_data = {name: value for name, value in parsed_data.items() if name in declared}
        dropped = sorted(set(parsed_data) - set(safe_data))
        if dropped:
            logger.debug(f"Ignored undeclared fields for {response_class.__name__}: {dropped}")
        return response_class(**safe_data)

    def _require_field(self, value, field_name: str, endpoint: str):
        """Guard against an API response that parsed but carries no result.

        Without this, a malformed or unexpected payload silently produces an
        object whose main attribute is ``None``, and the caller has no way to
        tell that apart from a legitimate empty result.
        """
        if value is None:
            raise RequestError(
                f"API response for {endpoint} did not contain '{field_name}'. "
                f"This usually means the API returned an unexpected shape.",
                endpoint=endpoint,
            )
        return value

    async def _request(self, endpoint: str, params: dict, extra_headers: Optional[dict] = None) -> dict:
        """Make an API request with error handling.

        Raises:
            DownloadInProgressError: On HTTP 202 (same video already queued).
            AuthenticationError: On 401/403.
            ClientError: On other 4xx (not retried).
            RequestError: On 5xx or network failure (retried by callers).
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        url = f"{self.BASE_URL}/{endpoint}"
        headers = self._get_headers()
        if self._cookies:
            headers["X-Cookies"] = self._cookies
        if extra_headers:
            headers.update(extra_headers)
        logger.info(f"Requesting: {url} with params: {params}")

        try:
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status == 202:
                    try:
                        data = await response.json()
                    except Exception:
                        data = {}
                    if not isinstance(data, dict):
                        data = {}

                    raise DownloadInProgressError(QueuedJobResponse(
                        job_id=data.get("job_id"),
                        video_id=data.get("video_id"),
                        status=data.get("status", JobStatus.PROCESSING.value),
                        message=data.get("message"),
                    ))

                data = await validate_rapidapi_response(
                    response,
                    AuthenticationError,
                    RequestError,
                    ClientError,
                )

                if not isinstance(data, dict):
                    return data

                return data

        except (AuthenticationError, RequestError, ClientError):
            raise
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            raise RequestError(
                f"Network error: {str(e)}",
                endpoint=endpoint,
                original_error=e
            )

    async def download_file(self, url: str, file_path: str, max_retries: Optional[int] = None,
                            retry_delay: Optional[float] = None) -> DownloadResult:
        """Download file from URL to local path with retry support"""
        retries = max_retries if max_retries is not None else self._max_retries
        delay = retry_delay if retry_delay is not None else self._retry_delay

        result_path = await _download_file(
            url=url, output_path=file_path,
            max_retries=retries, retry_delay=delay,
            show_progress=True, session=self._session
        )
        if result_path is None:
            raise DownloadError(f"Failed to download from {url}")
        return DownloadResult(file_path=result_path, file_size=Path(result_path).stat().st_size)

    def _extract_error_message(self, error_text: str) -> str:
        """Extract error message from API error response"""
        try:
            error_data = json.loads(error_text)
            if isinstance(error_data, dict):
                if "message" in error_data:
                    return error_data["message"]
                if "messages" in error_data:
                    return error_data["messages"]
        except Exception:
            pass
        return error_text

    # ==================== Public Methods ====================

    @with_retry(max_attempts=3, delay=1.0)
    async def video_info(self, url: str, cookies: Optional[str] = None) -> VideoInfoResponse:
        """
        Get detailed metadata for any supported video URL.

        Supports: YouTube, TikTok, Instagram, Facebook, SoundCloud, and more.
        Results are cached for 1 hour.

        Args:
            url: Full video URL from any supported site
            cookies: Optional Netscape cookies string for this request
        """
        extra_headers = {"X-Cookies": cookies} if cookies else None
        data = await self._request("video_info", {"video_url": url}, extra_headers=extra_headers)
        return self._parse_response_data(data, VideoInfoResponse)

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_to_server(
        self,
        video_id: str,
        file_format: str = "mp4",
        quality: str = "best",
        webhook_url: Optional[str] = None,
        format: Optional[str] = None,  # Deprecated: use file_format
        cookies: Optional[str] = None,
    ) -> ServerResponse:
        """
        Download YouTube video to server and get a temporary download URL (valid 10 min).

        Args:
            video_id: YouTube video ID (10-12 chars)
            file_format: Output format — "mp4" (video), "m4a" (audio), "mp3" (audio). Default: "mp4"
            quality: Output quality — "best", "worst", "1080p", "720p", "480p", "360p", "256k", "128k". Default: "best"
            webhook_url: Optional URL to receive POST callback when download completes
            format: Deprecated. Use file_format instead ("audio" → "m4a", "video" → "mp4")
            cookies: Optional Netscape cookies string for this request
        """
        params = {"video_id": video_id, "file_format": file_format, "quality": quality}
        if format is not None:
            params["format"] = format
        if webhook_url is not None:
            params["webhook_url"] = webhook_url
        extra_headers = {"X-Cookies": cookies} if cookies else None
        data = await self._request("youtube_to_server", params, extra_headers=extra_headers)
        response = self._parse_response_data(data, ServerResponse)
        self._require_field(response.download_url, "download_url", "youtube_to_server")
        return response

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_to_telegram(
        self,
        video_id: str,
        file_format: str = "m4a",
        quality: str = "best",
        webhook_url: Optional[str] = None,
        format: Optional[str] = None,  # Deprecated: use file_format
        cookies: Optional[str] = None,
    ) -> TelegramResponse:
        """
        Upload YouTube video/audio to Telegram and get the message link.

        Args:
            video_id: YouTube video ID (10-12 chars)
            file_format: Output format — "m4a" (audio, cached), "mp3" (audio, cached), "mp4" (video, cached). Default: "m4a"
            quality: Output quality — "best", "worst", "1080p", "720p", "480p", "360p", "256k", "128k". Default: "best"
            webhook_url: Optional URL to receive POST callback when upload completes
            format: Deprecated. Use file_format instead ("audio" → "m4a", "video" → "mp4")
            cookies: Optional Netscape cookies string for this request

        Raises:
            DownloadInProgressError: If the same video/format/quality is already
                being processed by another request (HTTP 202).
        """
        params = {"video_id": video_id, "file_format": file_format, "quality": quality}
        if format is not None:
            params["format"] = format
        if webhook_url is not None:
            params["webhook_url"] = webhook_url
        extra_headers = {"X-Cookies": cookies} if cookies else None
        data = await self._request("youtube_to_telegram", params, extra_headers=extra_headers)
        response = self._parse_response_data(data, TelegramResponse)
        self._require_field(response.file_url, "file_url", "youtube_to_telegram")
        return response

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_to_google(
        self,
        video_id: str,
        file_format: str = "mp4",
        quality: str = "best",
        webhook_url: Optional[str] = None,
        format: Optional[str] = None,  # Deprecated: use file_format
        cookies: Optional[str] = None,
    ) -> GoogleResponse:
        """
        Upload YouTube video or audio to Google Photos and get direct streaming/download URLs.

        Audio formats (mp3, m4a) are uploaded to Google Photos and return both `video_url`
        (from Google Photos) and `audio_url` (direct audio stream).

        Args:
            video_id: YouTube video ID (10-12 chars)
            file_format: Output format — "mp4" (video), "m4a" (audio), "mp3" (audio). Default: "mp4"
            quality: Output quality — "best", "worst", "1080p", "720p", "480p", "360p", "256k", "128k". Default: "best"
            webhook_url: Optional URL to receive POST callback when upload completes
            format: Deprecated. Use file_format instead ("audio" → "m4a", "video" → "mp4")
            cookies: Optional Netscape cookies string for this request

        Returns:
            GoogleResponse with video_url and optional audio_url.
        """
        params = {"video_id": video_id, "file_format": file_format, "quality": quality}
        if format is not None:
            params["format"] = format
        if webhook_url is not None:
            params["webhook_url"] = webhook_url
        extra_headers = {"X-Cookies": cookies} if cookies else None
        data = await self._request("youtube_to_google", params, extra_headers=extra_headers)
        response = self._parse_response_data(data, GoogleResponse)
        self._require_field(response.video_url, "video_url", "youtube_to_google")
        return response

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_live_hls(self, video_id: str, audio_only: bool = False) -> LiveStream:
        """
        Get HLS live stream playback URL for a YouTube Live (valid 10 min).

        Args:
            video_id: YouTube live video ID (10-12 chars)
            audio_only: Stream audio track only (no video). Default: False
        """
        data = await self._request(
            "youtube_live_hls",
            {"video_id": video_id, "audio_only": audio_only},
        )
        stream = LiveStream(url=data.get("url"), warning=data.get("warning"))
        self._require_field(stream.url, "url", "youtube_live_hls")
        return stream

    @with_retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, limit: int = 10) -> List[VideoSearchResult]:
        """
        Search YouTube videos, or resolve a video URL to its metadata.

        Args:
            query: Search query string, or a full URL (``http://`` /
                ``https://`` prefixed — without a scheme it is treated as text).
            limit: Maximum number of results (1-50). Ignored for URL queries.

        Returns:
            List of VideoSearchResult. A URL query yields a single-element list
            built from the ``/video_info`` response shape, with empty metadata
            fields that ``/video_info`` does not provide.
        """
        data = await self._request("search", {"query": query, "limit": limit})

        if isinstance(data, dict):
            # URL query: the API answers with one /video_info-shaped object.
            return [self._parse_response_data(data, VideoSearchResult)]

        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            results.append(self._parse_response_data(item, VideoSearchResult))
        return results