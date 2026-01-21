from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    RequestError,
    DownloadError as BaseDownloadError,
    APIConfig,
    with_retry,
)

logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

class DownloadError(BaseDownloadError):
    """Error when download fails"""
    def __init__(self, reason: str):
        super().__init__(f"Download failed: {reason}")
        self.reason = reason


class APIResponseError(RequestError):
    """Error when API doesn't return 200 status or returns an error message"""
    def __init__(self, message: str):
        super().__init__(f"API Error: {message}")
        self.message = message


# ==================== Data Models ====================

@dataclass
class DownloadProgress(BaseResponse):
    """
    Progress information for background download jobs.

    Returned by get_download_progress() endpoint.
    """
    success: bool = True
    job_id: str = None
    status: str = None  # initializing, downloading, background_processing, completed, failed
    video_id: str = None
    percentage: float = 0.0
    timestamp: float = None
    # Fields for 'downloading' status
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    speed: Optional[float] = None  # bytes/sec
    eta: Optional[int] = None  # seconds
    # Fields for 'completed' status
    result: Optional[dict] = None
    # Fields for 'failed' status
    error: Optional[str] = None
    # Fields for 'background_processing' status
    message: Optional[str] = None


@dataclass
class BackgroundJobResponse(BaseResponse):
    """
    Response when download is moved to background (HTTP 202).

    This happens when video is too large and would exceed timeout.
    Use job_id to track progress via get_download_progress().
    """
    success: bool = True
    status: str = "background_processing"
    job_id: str = None
    message: str = None
    try_after: int = None  # seconds
    progress_url: str = None
    video_id: str = None


@dataclass
class Channel(BaseResponse):
    """Channel information"""
    name: str = None
    id: str = None
    thumbnails: List[dict] = None
    link: str = None

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class Video(BaseResponse):
    """Base video object shared across multiple responses"""
    success: bool = False
    video_title: str = None
    video_id: str = None
    video_url: str = None
    thumbnail: str = None
    view_count: int = None
    duration: int = None
    description: str = None
    category: str = None
    tags: List[str] = None
    uploader: Channel = None

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

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class VideoInfoResponse(BaseResponse):
    """Detailed response for /video-info endpoint"""
    success: bool = False
    title: str = None
    description: str = None
    duration_seconds: int = None
    duration_string: str = None
    upload_date: str = None
    view_count: int = None
    concurrent_view_count: Optional[int] = None
    thumbnail: str = None
    language: Optional[str] = None
    has_subtitles: bool = False
    subtitle_languages: List[str] = None
    uploader_info: dict = None
    id: str = None
    webpage_url: str = None
    webpage_url_domain: str = None
    formats: List[dict] = None

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class TelegramResponse(BaseResponse):
    """Response when uploading YouTube video to Telegram"""
    success: bool = True
    file_url: str = None
    message_id: int = None
    chat_username: str = None
    # Progress tracking fields (added 2026-01-19)
    job_id: Optional[str] = None
    status: Optional[str] = None  # completed, background_processing
    progress_url: Optional[str] = None

    # to_dict() and to_json() inherited from BaseResponse




@dataclass
class DownloadResult(BaseResponse):
    """Represents result of a local file download"""
    file_path: str
    file_size: int

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class LiveStream(BaseResponse):
    """Represents a live stream (HLS/MP4)"""
    url: str

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class ServerDownloadField(BaseResponse):
    """
    DEPRECATED: This class is deprecated as of API v26.01.21+
    The download_url is now directly in ServerResponse root level.
    This class is kept only for reference and will be removed in future versions.
    """
    download_url: str
    filepath: Optional[str] = None


@dataclass
class Thumbnail(BaseResponse):
    """Thumbnail information"""
    url: str = None
    width: int = None
    height: int = None


@dataclass
class ViewCount(BaseResponse):
    """View count information"""
    text: str = None
    short: str = None


@dataclass
class Accessibility(BaseResponse):
    """Accessibility information"""
    title: str = None
    duration: str = None


@dataclass
class VideoSearchResult(BaseResponse):
    """Single video result"""
    type: str = None
    id: str = None
    title: str = None
    publishedTime: str = None
    duration: str = None
    viewCount: ViewCount = None
    thumbnails: List[Thumbnail] = None
    richThumbnail: Optional[dict] = None
    descriptionSnippet: List[dict] = None
    channel: Channel = None
    accessibility: Accessibility = None
    link: str = None
    shelfTitle: Optional[str] = None


@dataclass
class ServerResponse(Video):
    """Response for /youtube_to_server

    API v26.01.21+: Response structure changed to include download_url directly in root level.
    The nested download object has been removed.
    """
    download_url: Optional[str] = None
    _api_instance: Optional[YouTubeAPI] = None
    # Progress tracking fields (added 2026-01-19)
    job_id: Optional[str] = None
    status: Optional[str] = None  # completed, background_processing
    progress_url: Optional[str] = None

    async def download_file(self, file_path: str, max_retries: Optional[int] = None,
                            retry_delay: Optional[float] = None) -> DownloadResult:
        """
        Download the video file from the server URL to a local path.

        Args:
            file_path: Local path where the file will be saved
            max_retries: Number of retry attempts (default: instance default)
            retry_delay: Delay between retries in seconds (default: instance default)

        Returns:
            DownloadResult: Information about the downloaded file

        Raises:
            DownloadError: If download fails or no URL is available
        """
        if not self._api_instance:
            raise DownloadError("API instance not available")

        if not self.download_url:
            raise DownloadError("No download URL available")

        logger.info(f"Downloading file to {file_path} from server...")

        # Use instance defaults if not specified
        retries = max_retries if max_retries is not None else self._api_instance._max_retries
        delay = retry_delay if retry_delay is not None else self._api_instance._retry_delay

        return await self._api_instance.download_file(self.download_url, file_path, retries, delay)




# ==================== API Client ====================

class YouTubeAPI(BaseRapidAPI):
    """
    API client wrapper for YouTube to Telegram/Server/Host endpoints.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with YouTubeAPI(api_key="key") as client:
            result = await client.youtube_to_telegram("https://youtube.com/watch?v=...")
            print(result.video_title)

            # Use with config
            config = APIConfig(api_key="key", timeout=300, max_retries=5)
            async with YouTubeAPI(config=config) as client:
                result = await client.youtube_to_telegram("url")
    """

    BASE_URL = "https://youtube-to-telegram-uploader-api.p.rapidapi.com"
    DEFAULT_HOST = "youtube-to-telegram-uploader-api.p.rapidapi.com"

    def __init__(self, api_key: str, timeout: int = 300, max_retries: int = 5, retry_delay: float = 1.0,
                 max_wait_time: int = 0, config: Optional[APIConfig] = None):
        # Call parent __init__
        super().__init__(api_key=api_key, timeout=timeout, config=config)

        # Store additional YouTube-specific config
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._max_wait_time = max_wait_time

    # __aenter__ and __aexit__ inherited from BaseRapidAPI

    def _parse_response_data(self, data: dict, response_class):
        """Helper to convert API response into dataclass objects"""

        parsed_data = data.copy()  # Copy to avoid mutating original data

        # Handle nested objects depending on response class
        if response_class == ServerResponse:
            # API v26.01.21+: download_url is directly in root, no nested download object
            parsed_data['_api_instance'] = self

        elif response_class == TelegramResponse:
            # Flatten nested 'telegram' object if present
            if 'telegram' in parsed_data and isinstance(parsed_data['telegram'], dict):
                telegram_data = parsed_data.pop('telegram')
                # Merge telegram fields into root level
                parsed_data.update(telegram_data)

        elif response_class == VideoSearchResult:
            # Handle search result nested objects
            if 'viewCount' in parsed_data and isinstance(parsed_data['viewCount'], dict):
                parsed_data['viewCount'] = ViewCount(**parsed_data['viewCount'])

            if 'thumbnails' in parsed_data and isinstance(parsed_data['thumbnails'], list):
                parsed_data['thumbnails'] = [Thumbnail(**thumb) for thumb in parsed_data['thumbnails']]

            if 'channel' in parsed_data and isinstance(parsed_data['channel'], dict):
                parsed_data['channel'] = Channel(**parsed_data['channel'])

            if 'accessibility' in parsed_data and isinstance(parsed_data['accessibility'], dict):
                parsed_data['accessibility'] = Accessibility(**parsed_data['accessibility'])

        # Handle common Video fields present in Video-related responses only
        # Skip for TelegramResponse (which only has file_url, message_id, etc.)
        if response_class != TelegramResponse:
            if 'uploader' in parsed_data and isinstance(parsed_data['uploader'], dict):
                uploader_data = parsed_data['uploader'].copy()
                if 'channel_name' in uploader_data:
                    uploader_data['name'] = uploader_data.pop('channel_name')
                if 'channel_url' in uploader_data:
                    uploader_data['link'] = uploader_data.pop('channel_url')
                parsed_data['uploader'] = Channel(**uploader_data)

        # Normalize tags to list (only for Video-related responses)
        if response_class != TelegramResponse:
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

        try:
            return response_class(**parsed_data)
        except TypeError as e:
            logger.warning(f"Failed to create {response_class.__name__} with data: {parsed_data}")
            logger.warning(f"Error: {e}")
            # Return a safe instance with default values
            safe_data = {}
            for field_name, field_info in response_class.__dataclass_fields__.items():
                safe_data[field_name] = parsed_data.get(field_name,
                                                        field_info.default if field_info.default is not None else None)
            return response_class(**safe_data)

    async def _request(self, endpoint: str, params: dict) -> dict:
        """Make an API request with error handling and try_after support"""
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        url = f"{self.BASE_URL}/{endpoint}"
        headers = self._get_headers()
        logger.info(f"Requesting: {url} with params: {params}")

        try:
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status in (401, 403):
                    raise AuthenticationError(
                        "Authentication failed",
                        status_code=response.status,
                        endpoint=endpoint
                    )

                # Handle 202 Accepted - Background processing
                if response.status == 202:
                    try:
                        data = await response.json()
                        # Return BackgroundJobResponse info for the caller to handle
                        logger.info(f"Background processing initiated: job_id={data.get('job_id')}")
                        return data
                    except Exception:
                        error_text = await response.text()
                        raise RequestError(
                            error_text or "Background processing initiated but response parsing failed",
                            status_code=response.status,
                            endpoint=endpoint,
                            response_text=error_text
                        )

                if response.status != 200:
                    error_text = await response.text()
                    clean_message = self._extract_error_message(error_text)
                    raise RequestError(
                        clean_message,
                        status_code=response.status,
                        endpoint=endpoint
                    )

                try:
                    data = await response.json()
                except Exception:
                    text_response = await response.text()
                    raise RequestError(
                        text_response or "Empty response from API",
                        status_code=response.status,
                        endpoint=endpoint
                    )

                # Check if response is a dict before accessing dict methods
                if isinstance(data, dict):
                    try_after = data.get("try_after")

                    if try_after is not None:
                        if try_after > self._max_wait_time:
                            error_msg = data.get("message", f"Download delay required: {try_after} seconds")
                            raise RequestError(
                                error_msg,
                                status_code=response.status,
                                endpoint=endpoint
                            )

                        logger.info(f"Waiting {try_after} seconds before retry...")
                        await asyncio.sleep(try_after)

                        logger.info(f"Retrying request after waiting...")
                        async with self._session.get(url, headers=headers, params=params) as retry_response:
                            if retry_response.status != 200:
                                error_text = await retry_response.text()
                                clean_message = self._extract_error_message(error_text)
                                raise RequestError(
                                    clean_message,
                                    status_code=retry_response.status,
                                    endpoint=endpoint
                                )

                            data = await retry_response.json()

                    if data.get("success", False) is False:
                        error_msg = data.get("message") or data.get("messages", "Unknown error")
                        raise RequestError(
                            error_msg,
                            endpoint=endpoint
                        )

                return data

        except (AuthenticationError, RequestError):
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
        if not self._session:
            raise DownloadError("Session not initialized")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Use instance defaults if not specified
        retries = max_retries if max_retries is not None else self._max_retries
        delay = retry_delay if retry_delay is not None else self._retry_delay

        last_error = None

        for attempt in range(retries):
            try:
                logger.info(f"[Download] Starting download from: {url} (Attempt {attempt + 1}/{retries})")
                async with self._session.get(url) as response:
                    if response.status != 200:
                        raise DownloadError(f"HTTP {response.status}: Download failed")

                    total_size = int(response.headers.get('content-length', 0))

                    with open(file_path, "wb") as f:
                        downloaded = 0
                        last_logged = 0

                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                percentage = (downloaded / total_size) * 100
                                print(f"\r[Download] Progress: {percentage:.1f}%", end="", flush=True)

                                current_milestone = int(percentage // 10) * 10
                                if current_milestone > last_logged and current_milestone > 0:
                                    self._log_progress_checkpoint(current_milestone, "% completed")
                                    last_logged = current_milestone
                            else:
                                print(f"\r[Download] Downloaded: {downloaded:,} bytes", end="", flush=True)
                                current_mb = downloaded // (1024 * 1024)
                                if current_mb > last_logged:
                                    self._log_progress_checkpoint(current_mb, "MB downloaded")
                                    last_logged = current_mb

                    print()
                    logger.info(f"[Download] Success - {file_path} ({downloaded:,} bytes)")
                    return DownloadResult(file_path=file_path, file_size=downloaded)

            except DownloadError as e:
                last_error = e
                if attempt < retries - 1:
                    logger.warning(f"[Download] Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[Download] All {retries} attempts failed")
                    raise
            except Exception as e:
                last_error = DownloadError(f"Download error: {str(e)}")
                if attempt < retries - 1:
                    logger.warning(f"[Download] Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[Download] All {retries} attempts failed")
                    raise last_error

        # This should not be reached, but just in case
        if last_error:
            raise last_error
        raise DownloadError("Download failed for unknown reason")

    def _log_progress_checkpoint(self, value: int, unit: str):
        """Helper method to log progress checkpoints"""
        print()
        logger.info(f"[Download] {value}{unit}")

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

    async def _wait_for_background_job(self, job_id: str, poll_interval: float = 2.0,
                                       max_wait_time: Optional[int] = None) -> dict:
        """
        Wait for a background job to complete by polling the progress endpoint.

        Args:
            job_id: The job ID to track
            poll_interval: Time in seconds between progress checks (default: 2.0)
            max_wait_time: Maximum time to wait in seconds (default: None = no limit)

        Returns:
            dict: The completed job result

        Raises:
            RequestError: If job fails or max wait time exceeded
        """
        start_time = asyncio.get_event_loop().time() if max_wait_time else None
        logger.info(f"[BACKGROUND_WAIT] Waiting for job {job_id} to complete...")

        while True:
            # Check if max wait time exceeded
            if max_wait_time and start_time:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_wait_time:
                    raise RequestError(
                        f"Background job timeout: exceeded {max_wait_time}s",
                        endpoint="download_progress"
                    )

            # Get progress
            try:
                progress = await self.get_download_progress(job_id)
            except Exception as e:
                logger.error(f"[BACKGROUND_WAIT] Error checking progress: {e}")
                raise RequestError(
                    f"Failed to check background job progress: {str(e)}",
                    endpoint="download_progress",
                    original_error=e
                )

            status = progress.status
            logger.info(f"[BACKGROUND_WAIT] Job {job_id} status: {status} ({progress.percentage:.1f}%)")

            if status == "completed":
                logger.info(f"[BACKGROUND_WAIT] Job {job_id} completed successfully")
                return progress.result

            elif status == "failed":
                error_msg = progress.error or "Unknown error"
                logger.error(f"[BACKGROUND_WAIT] Job {job_id} failed: {error_msg}")
                raise RequestError(
                    f"Background job failed: {error_msg}",
                    endpoint="download_progress"
                )

            elif status in ("initializing", "downloading", "background_processing"):
                # Still processing, wait and retry
                await asyncio.sleep(poll_interval)
                continue

            else:
                # Unknown status
                logger.warning(f"[BACKGROUND_WAIT] Unknown job status: {status}")
                await asyncio.sleep(poll_interval)
                continue

    # ==================== Public Methods ====================

    @with_retry(max_attempts=3, delay=1.0)
    async def video_info(self, url: str) -> VideoInfoResponse:
        """Get detailed video info from YouTube URL or video ID"""
        data = await self._request("video_info", {"video_url": url})
        return self._parse_response_data(data, VideoInfoResponse)

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_to_server(self, video_id: str, format: str = "audio",
                               auto_wait: bool = True, poll_interval: float = 2.0,
                               max_wait_time: Optional[int] = None) -> ServerResponse:
        """
        Get downloadable server URL for a given video.

        Args:
            video_id: YouTube video ID or URL
            format: Format type ("audio" or "video")
            auto_wait: If True, automatically wait for background jobs to complete (default: True)
            poll_interval: Time in seconds between progress checks when waiting (default: 2.0)
            max_wait_time: Maximum time to wait for background jobs in seconds (default: None = no limit)

        Returns:
            ServerResponse: Response containing download URL and video info
                          (API v26.01.21+: download_url is directly in root level)

        Example:
            async with YouTubeAPI(api_key="key") as client:
                # Auto-wait enabled (default)
                result = await client.youtube_to_server("video_id")

                # Access download URL
                print(result.download_url)

                # Download the file
                await result.download_file("/path/to/save.mp4")

                # Disable auto-wait to handle background jobs manually
                result = await client.youtube_to_server("video_id", auto_wait=False)
                if result.status == "background_processing":
                    # Handle background job manually using result.job_id
                    progress = await client.get_download_progress(result.job_id)
        """
        data = await self._request("youtube_to_server", {"video_id": video_id, "format": format})

        # Check if response indicates background processing
        if auto_wait and data.get("status") == "background_processing" and data.get("job_id"):
            job_id = data["job_id"]
            logger.info(f"[AUTO_WAIT] Background job detected (job_id={job_id}), waiting for completion...")

            # Wait for job to complete
            result = await self._wait_for_background_job(job_id, poll_interval, max_wait_time)

            # Parse the final result
            return self._parse_response_data(result, ServerResponse)

        return self._parse_response_data(data, ServerResponse)

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_to_telegram(self, video_id: str, format: str = "audio",
                                  auto_wait: bool = True, poll_interval: float = 2.0,
                                  max_wait_time: Optional[int] = None) -> TelegramResponse:
        """
        Upload YouTube video to Telegram.

        Args:
            video_id: YouTube video ID or URL
            format: Format type ("audio" or "video")
            auto_wait: If True, automatically wait for background jobs to complete (default: True)
            poll_interval: Time in seconds between progress checks when waiting (default: 2.0)
            max_wait_time: Maximum time to wait for background jobs in seconds (default: None = no limit)

        Returns:
            TelegramResponse: Response containing Telegram file URL and message info

        Example:
            async with YouTubeAPI(api_key="key") as client:
                # Auto-wait enabled (default)
                result = await client.youtube_to_telegram("video_id")
                print(result.file_url)

                # Disable auto-wait to handle background jobs manually
                result = await client.youtube_to_telegram("video_id", auto_wait=False)
                if result.status == "background_processing":
                    # Handle background job manually using result.job_id
                    progress = await client.get_download_progress(result.job_id)
        """
        data = await self._request("youtube_to_telegram", {"video_id": video_id, "format": format})

        # Check if response indicates background processing
        if auto_wait and data.get("status") == "background_processing" and data.get("job_id"):
            job_id = data["job_id"]
            logger.info(f"[AUTO_WAIT] Background job detected (job_id={job_id}), waiting for completion...")

            # Wait for job to complete
            result = await self._wait_for_background_job(job_id, poll_interval, max_wait_time)

            # Parse the final result
            return self._parse_response_data(result, TelegramResponse)

        return self._parse_response_data(data, TelegramResponse)

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_live_hls(self, video_id: str) -> LiveStream:
        """Get HLS live stream URL for a YouTube Live"""
        data = await self._request("youtube_live_hls", {"video_id": video_id})
        return LiveStream(url=data.get("url"))

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_live_mp4(self, video_id: str) -> LiveStream:
        """Get MP4 live stream URL for a YouTube Live"""
        data = await self._request("youtube_live_mp4", {"video_id": video_id})
        return LiveStream(url=data.get("url"))

    @with_retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, limit: int = 5) -> List[VideoSearchResult]:
        """Search on YouTube for a given query"""
        data = await self._request("search", {"query": query, "limit": limit})
        results = []
        if isinstance(data, list):
            for item in data:
                results.append(self._parse_response_data(item, VideoSearchResult))
        return results

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_video_stream(self, video_id: str, format: str = "audio") -> LiveStream:
        """Get MP4 live stream URL for a YouTube video"""
        data = await self._request("youtube_video_stream", {"video_id": video_id, "format": format})
        return LiveStream(url=data.get("url"))

    @with_retry(max_attempts=3, delay=1.0)
    async def get_download_progress(self, job_id: str) -> DownloadProgress:
        """
        Get download progress for a background job.

        Use this endpoint to check the progress of downloads that were moved
        to background processing (when you receive a BackgroundJobResponse).

        Args:
            job_id: The job ID returned from youtube_to_server or youtube_to_telegram

        Returns:
            DownloadProgress: Progress information including status, percentage, ETA, etc.

        Example:
            async with YouTubeAPI(api_key="key") as client:
                try:
                    result = await client.youtube_to_server("video_id")
                except RequestError as e:
                    # If status code is 202, download is in background
                    if e.status_code == 202:
                        job_id = e.response_text.get("job_id")
                        # Check progress
                        progress = await client.get_download_progress(job_id)
                        print(f"Progress: {progress.percentage}%")

                        if progress.status == "completed":
                            result = progress.result  # Full result data
        """
        data = await self._request("download_progress", {"job_id": job_id})

        # Parse into DownloadProgress dataclass
        return DownloadProgress(
            success=data.get("success", True),
            job_id=data.get("job_id"),
            status=data.get("status"),
            video_id=data.get("video_id"),
            percentage=data.get("percentage", 0.0),
            timestamp=data.get("timestamp"),
            downloaded_bytes=data.get("downloaded_bytes"),
            total_bytes=data.get("total_bytes"),
            speed=data.get("speed"),
            eta=data.get("eta"),
            result=data.get("result"),
            error=data.get("error"),
            message=data.get("message")
        )