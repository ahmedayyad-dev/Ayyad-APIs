import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import aiohttp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ==================== Custom Exceptions ====================

class DownloadError(Exception):
    """Error when download fails"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Download failed: {reason}")


class APIResponseError(Exception):
    """Error when API doesn't return 200 status or returns an error message"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(f"API Error: {message}")


# ==================== Data Models ====================

@dataclass
class Channel:
    """Represents a video uploader channel"""
    name: str = None
    url: str = None


@dataclass
class Video:
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


@dataclass
class VideoInfoResponse:
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


@dataclass
class TelegramResponse(Video):
    """Response when uploading YouTube video to Telegram"""
    file_url: str = None
    message_id: int = None
    chat_username: str = None


@dataclass
class TelegramInfoResponse(Video):
    """Response for YouTube video with Telegram info"""
    telegram: Optional[TelegramResponse] = None


@dataclass
class DownloadResult:
    """Represents result of a local file download"""
    file_path: str
    file_size: int


@dataclass
class LiveStream:
    """Represents a live stream (HLS/MP4)"""
    url: str


@dataclass
class ServerDownloadField:
    """Download field returned in youtube_to_server endpoint"""
    download_url: str


@dataclass
class ServerResponse(Video):
    """Response for /youtube_to_server"""
    download: ServerDownloadField = None
    _api_instance: Optional['YouTubeAPI'] = None

    async def download_file(self, file_path: str, max_retries=None, retry_delay=None) -> DownloadResult:
        if not self._api_instance:
            raise DownloadError("API instance not available")
        logger.info(f"Downloading file to {file_path} from server...")
        return await self._api_instance.download_file(self.download.download_url, file_path, max_retries, retry_delay)


@dataclass
class HostDownloadField:
    """Download field returned in youtube_to_host endpoint"""
    download_url: str


@dataclass
class HostResponse(Video):
    """Response for /youtube_to_host"""
    hosting: HostDownloadField = None
    _api_instance: Optional['YouTubeAPI'] = None

    async def download_file(self, file_path: str, max_retries=None, retry_delay=None) -> DownloadResult:
        if not self._api_instance:
            raise DownloadError("API instance not available")
        logger.info(f"Downloading file to {file_path} from hosting...")
        return await self._api_instance.download_file(self.hosting.download_url, file_path, max_retries, retry_delay)


# ==================== API Client ====================

class YouTubeAPI:
    """API client wrapper for YouTube to Telegram/Server/Host endpoints"""

    def __init__(self, api_key: str, timeout: int = 300, max_retries: int = 5, retry_delay: float = 1.0):
        self.api_key = api_key
        self._base_url = "https://youtube-to-telegram-uploader-api.p.rapidapi.com"
        self._headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "youtube-to-telegram-uploader-api.p.rapidapi.com"
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    def _parse_response_data(self, data: dict, response_class):
        """Helper to convert API response into dataclass objects"""

        parsed_data = data.copy()  # Copy to avoid mutating original data

        # Handle nested objects depending on response class
        if response_class == TelegramInfoResponse:
            if 'telegram' in parsed_data and isinstance(parsed_data['telegram'], dict):
                telegram_data = parsed_data['telegram'].copy()
                # Handle uploader mapping inside telegram response
                if 'uploader' in telegram_data and isinstance(telegram_data['uploader'], dict):
                    uploader_data = telegram_data['uploader'].copy()
                    if 'channel_name' in uploader_data:
                        uploader_data['name'] = uploader_data.pop('channel_name')
                    if 'channel_url' in uploader_data:
                        uploader_data['url'] = uploader_data.pop('channel_url')
                    telegram_data['uploader'] = Channel(**uploader_data)
                parsed_data['telegram'] = TelegramResponse(**telegram_data)

        elif response_class == ServerResponse:
            if 'download' in parsed_data and isinstance(parsed_data['download'], dict):
                parsed_data['download'] = ServerDownloadField(**parsed_data['download'])
            parsed_data['_api_instance'] = self

        elif response_class == HostResponse:
            if 'hosting' in parsed_data and isinstance(parsed_data['hosting'], dict):
                parsed_data['hosting'] = HostDownloadField(**parsed_data['hosting'])
            parsed_data['_api_instance'] = self

        elif response_class == TelegramResponse:
            # No special handling needed
            pass

        # Handle common Video fields present in almost all responses
        if 'uploader' in parsed_data and isinstance(parsed_data['uploader'], dict):
            uploader_data = parsed_data['uploader'].copy()
            if 'channel_name' in uploader_data:
                uploader_data['name'] = uploader_data.pop('channel_name')
            if 'channel_url' in uploader_data:
                uploader_data['url'] = uploader_data.pop('channel_url')
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

        try:
            return response_class(**parsed_data)
        except TypeError as e:
            logger.warning(f"Failed to create {response_class.__name__} with data: {parsed_data}")
            logger.warning(f"Error: {e}")
            # Return a safe instance with default values
            safe_data = {}
            for field_name, field_info in response_class.__dataclass_fields__.items():
                safe_data[field_name] = parsed_data.get(field_name, field_info.default if field_info.default is not None else None)
            return response_class(**safe_data)

    async def _request(self, endpoint: str, params: dict) -> dict:
        """Make an API request with error handling"""
        url = f"{self._base_url}/{endpoint}"
        logger.info(f"Requesting: {url} with params: {params}")

        try:
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    clean_message = self._extract_error_message(error_text)
                    raise APIResponseError(clean_message)

                try:
                    data = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    text_response = await response.text()
                    raise APIResponseError(text_response or "Empty response from API")

                if isinstance(data, dict) and data.get("success", False) is False:
                    error_msg = data.get("message") or data.get("messages", "Unknown error")
                    raise APIResponseError(error_msg)

                return data

        except APIResponseError:
            raise
        except Exception as e:
            raise APIResponseError(f"Connection error: {str(e)}")

    async def download_file(self, url: str, file_path: str, max_retries: Optional[int] = None,
                            retry_delay: Optional[float] = None) -> DownloadResult:
        """Download file from URL to local path"""
        if not self._session:
            raise DownloadError("Session not initialized")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"[Download] Starting download from: {url}")
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
                                self._log_progress_checkpoint(current_milestone, "completed")
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

        except DownloadError:
            raise
        except Exception as e:
            raise DownloadError(f"Download error: {str(e)}")

    def _log_progress_checkpoint(self, value: int, unit: str):
        """Helper method to log progress checkpoints"""
        print()
        logger.info(f"[Download] {value}{unit}")

    def _extract_error_message(self, error_text: str) -> str:
        """Extract error message from API error response"""
        try:
            import json
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

    async def video_info(self, url: str) -> VideoInfoResponse:
        """Get detailed video info from YouTube URL or video ID"""
        data = await self._request("video_info", {"video_url": url})
        return self._parse_response_data(data, VideoInfoResponse)

    async def youtube_to_server(self, video_id: str, format: str = "audio") -> ServerResponse:
        """Get downloadable server URL for a given video"""
        params = {"video_id": video_id, "format": format}
        data = await self._request("youtube_to_server", params)
        return self._parse_response_data(data, ServerResponse)

    async def youtube_to_host(self, video_id: str, format: str = "audio") -> HostResponse:
        """Get hosted download URL for a given video"""
        params = {"video_id": video_id, "format": format}
        data = await self._request("youtube_to_host", params)
        return self._parse_response_data(data, HostResponse)

    async def youtube_to_telegram(self, video_id: str, format: str = "audio", mode: str = "telegram") -> TelegramInfoResponse:
        """Upload YouTube video to Telegram or fetch Telegram+video info"""
        params = {"video_id": video_id, "format": format, "mode": mode}
        data = await self._request("youtube_to_telegram", params)
        return self._parse_response_data(data, TelegramInfoResponse)

    async def youtube_live_hls(self, video_id: str) -> LiveStream:
        """Get HLS live stream URL for a YouTube video"""
        data = await self._request("youtube_live_hls", {"video_id": video_id})
        return LiveStream(url=data.get("url"))

    async def youtube_live_mp4(self, video_id: str) -> LiveStream:
        """Get MP4 live stream URL for a YouTube video"""
        data = await self._request("youtube_live_mp4", {"video_id": video_id})
        return LiveStream(url=data.get("url"))

