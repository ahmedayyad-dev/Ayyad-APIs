"""
AllTube CDN Extractor API wrapper for extracting video information from various platforms.

This module provides a simple async interface to interact with AllTube CDN Extractor API
through RapidAPI, allowing users to extract video information, download URLs, and formats
from various video platforms (YouTube, Facebook, Instagram, TikTok, etc.).

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import aiohttp

# Import shared download function
from ..utils import download_file


# Configure logging
logger = logging.getLogger(__name__)


# ==================== Custom Exceptions ====================

class AllTubeError(Exception):
    """Base exception for AllTube API errors."""
    pass


class AllTubeAuthenticationError(AllTubeError):
    """Raised when API authentication fails."""
    pass


class AllTubeRequestError(AllTubeError):
    """Raised when API request fails."""
    pass


class AllTubeInvalidURLError(AllTubeError):
    """Raised when video URL is invalid or malformed."""
    pass


# ==================== Data Models ====================

@dataclass
class Format:
    """Represents a video/audio format with quality and download information."""
    format_id: str
    ext: str
    url: str
    format_note: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    filesize: Optional[int] = None
    filesize_approx: Optional[int] = None
    tbr: Optional[float] = None
    vbr: Optional[float] = None
    abr: Optional[float] = None
    quality: Optional[int] = None
    audio_channels: Optional[int] = None
    protocol: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Format":
        """Create Format from API response dictionary."""
        return cls(
            format_id=data.get("format_id", ""),
            ext=data.get("ext", ""),
            url=data.get("url", ""),
            format_note=data.get("format_note"),
            width=data.get("width"),
            height=data.get("height"),
            resolution=data.get("resolution"),
            fps=data.get("fps"),
            vcodec=data.get("vcodec"),
            acodec=data.get("acodec"),
            filesize=data.get("filesize"),
            filesize_approx=data.get("filesize_approx"),
            tbr=data.get("tbr"),
            vbr=data.get("vbr"),
            abr=data.get("abr"),
            quality=data.get("quality"),
            audio_channels=data.get("audio_channels"),
            protocol=data.get("protocol")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format_id": self.format_id,
            "ext": self.ext,
            "url": self.url,
            "format_note": self.format_note,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "fps": self.fps,
            "vcodec": self.vcodec,
            "acodec": self.acodec,
            "filesize": self.filesize,
            "filesize_approx": self.filesize_approx,
            "tbr": self.tbr,
            "vbr": self.vbr,
            "abr": self.abr,
            "quality": self.quality,
            "audio_channels": self.audio_channels,
            "protocol": self.protocol
        }

    def is_video(self) -> bool:
        """Check if this format contains video."""
        return self.vcodec and self.vcodec != "none"

    def is_audio(self) -> bool:
        """Check if this format contains audio."""
        return self.acodec and self.acodec != "none"

    def is_video_only(self) -> bool:
        """Check if this format is video only (no audio)."""
        return self.is_video() and not self.is_audio()

    def is_audio_only(self) -> bool:
        """Check if this format is audio only (no video)."""
        return self.is_audio() and not self.is_video()

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        return_bytes: bool = False
    ) -> Union[bytes, str, None]:
        """
        Download this format's media file.

        Args:
            output_path: Path to save the file. If None, auto-generates filename.
            return_bytes: If True, returns bytes instead of saving to file.

        Returns:
            - bytes if return_bytes=True
            - str (file path) if saved to disk
            - None if download fails

        Example:
            best_format = video_info.get_best_format()
            path = await best_format.download("my_video.mp4")
        """
        if not self.url:
            logger.error("No download URL available")
            return None

        # Determine default filename and extension
        if self.is_video():
            default_filename = f"video_{self.format_id}"
        elif self.is_audio():
            default_filename = f"audio_{self.format_id}"
        else:
            default_filename = f"media_{self.format_id}"

        default_ext = f".{self.ext}" if self.ext else ".bin"

        return await download_file(
            url=self.url,
            output_path=output_path,
            return_bytes=return_bytes,
            default_filename=default_filename,
            default_ext=default_ext
        )


@dataclass
class Subtitle:
    """Represents a subtitle track."""
    url: str
    name: str
    ext: str
    language: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], language: str = "") -> "Subtitle":
        """Create Subtitle from API response dictionary."""
        return cls(
            url=data.get("url", ""),
            name=data.get("name", ""),
            ext=data.get("ext", ""),
            language=language
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "name": self.name,
            "ext": self.ext,
            "language": self.language
        }

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        return_bytes: bool = False
    ) -> Union[bytes, str, None]:
        """
        Download this subtitle file.

        Args:
            output_path: Path to save the file. If None, auto-generates from name and language.
            return_bytes: If True, returns bytes instead of saving to file.

        Returns:
            - bytes if return_bytes=True
            - str (file path) if saved to disk
            - None if download fails

        Example:
            subtitle = video_info.get_subtitles_by_language("en")[0]
            path = await subtitle.download("subtitles.srt")
        """
        if not self.url:
            logger.error("No subtitle URL available")
            return None

        # Generate filename from language and name
        if output_path is None and not return_bytes:
            safe_name = "".join(c for c in self.name if c.isalnum() or c in (' ', '-', '_'))
            safe_name = safe_name.strip()[:30] or 'subtitle'
            lang_prefix = f"{self.language}_" if self.language else ""
            output_path = f"{lang_prefix}{safe_name}"

        default_ext = f".{self.ext}" if self.ext else ".srt"

        return await download_file(
            url=self.url,
            output_path=output_path,
            return_bytes=return_bytes,
            default_filename="subtitle",
            default_ext=default_ext
        )


@dataclass
class Thumbnail:
    """Represents a video thumbnail with dimensions."""
    url: str
    id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[str] = None
    preference: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thumbnail":
        """Create Thumbnail from API response dictionary."""
        return cls(
            url=data.get("url", ""),
            id=str(data.get("id", "")),
            width=data.get("width"),
            height=data.get("height"),
            resolution=data.get("resolution"),
            preference=data.get("preference")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "preference": self.preference
        }

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        return_bytes: bool = False
    ) -> Union[bytes, str, None]:
        """
        Download this thumbnail image.

        Args:
            output_path: Path to save the file. If None, auto-generates from resolution.
            return_bytes: If True, returns bytes instead of saving to file.

        Returns:
            - bytes if return_bytes=True
            - str (file path) if saved to disk
            - None if download fails

        Example:
            best_thumb = video_info.get_best_thumbnail()
            path = await best_thumb.download("thumbnail.jpg")
        """
        if not self.url:
            logger.error("No thumbnail URL available")
            return None

        # Generate filename from resolution or id
        if output_path is None and not return_bytes:
            if self.resolution:
                output_path = f"thumbnail_{self.resolution}"
            elif self.id:
                output_path = f"thumbnail_{self.id}"

        return await download_file(
            url=self.url,
            output_path=output_path,
            return_bytes=return_bytes,
            default_filename="thumbnail",
            default_ext=".jpg"
        )


@dataclass
class VideoInfo:
    """Complete video information extracted from the API."""
    # Basic info
    title: str
    url: str
    webpage_url: Optional[str] = None

    # Video details
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[str] = None
    ext: Optional[str] = None

    # Content info
    description: Optional[str] = None
    thumbnail: Optional[str] = None

    # Channel/Uploader info
    uploader: Optional[str] = None
    uploader_id: Optional[str] = None
    channel_is_verified: Optional[bool] = None

    # Dates
    upload_date: Optional[str] = None
    release_date: Optional[str] = None

    # Video/Audio codecs
    vcodec: Optional[str] = None
    acodec: Optional[str] = None

    # Formats and quality
    format_id: Optional[str] = None
    formats: List[Format] = field(default_factory=list)
    requested_formats: List[Format] = field(default_factory=list)

    # Subtitles and thumbnails
    subtitles: Dict[str, List[Subtitle]] = field(default_factory=dict)
    thumbnails: List[Thumbnail] = field(default_factory=list)

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    is_live: Optional[bool] = None

    # Raw data for advanced usage
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoInfo":
        """Create VideoInfo from API response dictionary."""
        # Parse formats
        formats = []
        for fmt in data.get("formats", []):
            try:
                formats.append(Format.from_dict(fmt))
            except Exception as e:
                logger.debug(f"Failed to parse format: {e}")

        # Parse requested formats
        requested_formats = []
        for fmt in data.get("requested_formats", []):
            try:
                requested_formats.append(Format.from_dict(fmt))
            except Exception as e:
                logger.debug(f"Failed to parse requested format: {e}")

        # Parse subtitles
        subtitles = {}
        for lang, subs in data.get("subtitles", {}).items():
            subtitles[lang] = []
            for sub in subs:
                try:
                    subtitles[lang].append(Subtitle.from_dict(sub, language=lang))
                except Exception as e:
                    logger.debug(f"Failed to parse subtitle: {e}")

        # Parse thumbnails
        thumbnails = []
        for thumb in data.get("thumbnails", []):
            try:
                thumbnails.append(Thumbnail.from_dict(thumb))
            except Exception as e:
                logger.debug(f"Failed to parse thumbnail: {e}")

        return cls(
            title=data.get("title", data.get("fulltitle", "Unknown")),
            url=data.get("url", ""),
            webpage_url=data.get("webpage_url") or data.get("original_url"),
            duration=data.get("duration"),
            width=data.get("width"),
            height=data.get("height"),
            resolution=data.get("resolution"),
            ext=data.get("ext"),
            description=data.get("description"),
            thumbnail=data.get("thumbnail"),
            uploader=data.get("uploader"),
            uploader_id=data.get("uploader_id"),
            channel_is_verified=data.get("channel_is_verified"),
            upload_date=data.get("upload_date"),
            release_date=data.get("release_date"),
            vcodec=data.get("vcodec"),
            acodec=data.get("acodec"),
            format_id=data.get("format_id"),
            formats=formats,
            requested_formats=requested_formats,
            subtitles=subtitles,
            thumbnails=thumbnails,
            tags=data.get("tags", []),
            is_live=data.get("is_live"),
            raw_data=data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (simplified, without raw_data)."""
        return {
            "title": self.title,
            "url": self.url,
            "webpage_url": self.webpage_url,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "ext": self.ext,
            "description": self.description,
            "thumbnail": self.thumbnail,
            "uploader": self.uploader,
            "uploader_id": self.uploader_id,
            "channel_is_verified": self.channel_is_verified,
            "upload_date": self.upload_date,
            "release_date": self.release_date,
            "vcodec": self.vcodec,
            "acodec": self.acodec,
            "format_id": self.format_id,
            "formats": [fmt.to_dict() for fmt in self.formats],
            "requested_formats": [fmt.to_dict() for fmt in self.requested_formats],
            "subtitles": {
                lang: [sub.to_dict() for sub in subs]
                for lang, subs in self.subtitles.items()
            },
            "thumbnails": [thumb.to_dict() for thumb in self.thumbnails],
            "tags": self.tags,
            "is_live": self.is_live
        }

    def get_best_format(
        self,
        prefer_quality: str = "best",
        format_type: str = "combined"
    ) -> Optional[Format]:
        """
        Get the best format based on preferences.

        Args:
            prefer_quality: "best" or "worst" quality
            format_type: "combined" (video+audio), "video", or "audio"

        Returns:
            Best matching Format or None
        """
        if not self.formats:
            return None

        # Filter formats by type
        if format_type == "video":
            candidates = [f for f in self.formats if f.is_video() and not f.is_audio_only()]
        elif format_type == "audio":
            candidates = [f for f in self.formats if f.is_audio()]
        else:  # combined
            candidates = [f for f in self.formats if f.is_video() and f.is_audio()]

        if not candidates:
            return None

        # Sort by quality (using tbr - total bitrate)
        reverse = (prefer_quality == "best")
        sorted_formats = sorted(
            candidates,
            key=lambda f: (f.tbr or 0, f.height or 0, f.width or 0),
            reverse=reverse
        )

        return sorted_formats[0] if sorted_formats else None

    def get_format_by_resolution(self, resolution: str) -> Optional[Format]:
        """
        Get format by resolution (e.g., "1920x1080", "1280x720").

        Args:
            resolution: Resolution string like "1920x1080"

        Returns:
            Matching Format or None
        """
        for fmt in self.formats:
            if fmt.resolution == resolution:
                return fmt
        return None

    def get_best_thumbnail(self) -> Optional[Thumbnail]:
        """Get the highest quality thumbnail."""
        if not self.thumbnails:
            return None

        # Sort by preference (higher is better) and resolution
        sorted_thumbs = sorted(
            self.thumbnails,
            key=lambda t: (
                t.preference or -999,
                (t.width or 0) * (t.height or 0)
            ),
            reverse=True
        )

        return sorted_thumbs[0] if sorted_thumbs else None

    def get_subtitles_by_language(self, language: str) -> List[Subtitle]:
        """
        Get subtitles for a specific language.

        Args:
            language: Language code (e.g., "en", "ar")

        Returns:
            List of Subtitle objects for that language
        """
        return self.subtitles.get(language, [])

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        return_bytes: bool = False,
        prefer_quality: str = "best",
        format_type: str = "combined"
    ) -> Union[bytes, str, None]:
        """
        Download the video using the best available format.

        Args:
            output_path: Path to save the file. If None, generates from title.
            return_bytes: If True, returns bytes instead of saving to file.
            prefer_quality: "best" or "worst" quality
            format_type: "combined" (video+audio), "video", or "audio"

        Returns:
            - bytes if return_bytes=True
            - str (file path) if saved to disk
            - None if download fails

        Example:
            video_info = await client.get_info("https://youtube.com/watch?v=...")
            # Download best quality
            path = await video_info.download("my_video.mp4")
            # Download audio only
            path = await video_info.download("audio.mp3", format_type="audio")
        """
        # Get best format based on preferences
        best_format = self.get_best_format(prefer_quality=prefer_quality, format_type=format_type)

        if not best_format:
            logger.error("No suitable format found for download")
            return None

        # Generate filename from title if no output_path provided
        if output_path is None and not return_bytes:
            safe_title = "".join(c for c in self.title if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title.strip()[:50] or 'video'
            output_path = safe_title

        # Use the format's download method
        return await best_format.download(
            output_path=output_path,
            return_bytes=return_bytes
        )


# ==================== AllTube API Client ====================

class AllTubeAPI:
    """
    Async client for AllTube CDN Extractor API via RapidAPI.

    This API extracts video information from various platforms including:
    - YouTube
    - Facebook
    - Instagram
    - TikTok
    - Twitter/X
    - Vimeo
    - And many more...

    Example:
        async with AllTubeAPI(api_key="your_key") as client:
            # Get video information
            video_info = await client.get_info("https://www.youtube.com/watch?v=...")
            print(f"Title: {video_info.title}")
            print(f"Duration: {video_info.duration}s")

            # Get best quality format
            best = video_info.get_best_format(prefer_quality="best")
            if best:
                print(f"Best format: {best.resolution} - {best.url}")

            # Get subtitles
            subs = video_info.get_subtitles_by_language("en")
            for sub in subs:
                print(f"Subtitle: {sub.name} ({sub.ext})")
    """

    BASE_URL = "https://alltube-cdn-extractor.p.rapidapi.com"
    DEFAULT_HOST = "alltube-cdn-extractor.p.rapidapi.com"

    def __init__(
        self,
        api_key: str,
        rapidapi_host: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Initialize AllTube API client.

        Args:
            api_key: RapidAPI key for authentication
            rapidapi_host: RapidAPI host (defaults to alltube-cdn-extractor.p.rapidapi.com)
            timeout: Request timeout in seconds (default: 60)
        """
        self.api_key = api_key
        self.rapidapi_host = rapidapi_host or self.DEFAULT_HOST
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info("AllTube API client initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        logger.debug("HTTP session created")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            logger.debug("HTTP session closed")
        return False

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "x-rapidapi-host": self.rapidapi_host,
            "x-rapidapi-key": self.api_key
        }

    async def get_info(self, url: str) -> VideoInfo:
        """
        Extract video information from a URL.

        Args:
            url: Video URL from supported platforms (YouTube, Facebook, Instagram, etc.)

        Returns:
            VideoInfo object with complete video information

        Raises:
            AllTubeInvalidURLError: If URL is invalid
            AllTubeAuthenticationError: If authentication fails
            AllTubeRequestError: If request fails

        Example:
            async with AllTubeAPI(api_key="key") as client:
                info = await client.get_info("https://www.youtube.com/watch?v=...")
                print(info.title)
        """
        if not url or not isinstance(url, str):
            raise AllTubeInvalidURLError("URL must be a non-empty string")

        if not url.startswith(("http://", "https://")):
            raise AllTubeInvalidURLError("URL must start with http:// or https://")

        logger.info(f"Extracting video info from: {url}")

        if not self._session:
            raise AllTubeError("Session not initialized. Use async context manager.")

        endpoint = f"{self.BASE_URL}/getInfo"
        headers = self._get_headers()
        params = {"url": url}

        try:
            async with self._session.get(endpoint, headers=headers, params=params) as response:
                # Check for authentication errors
                if response.status == 401 or response.status == 403:
                    raise AllTubeAuthenticationError(
                        f"Authentication failed: {response.status}"
                    )

                # Check for other errors
                if response.status != 200:
                    error_text = await response.text()
                    raise AllTubeRequestError(
                        f"Request failed with status {response.status}: {error_text}"
                    )

                data = await response.json()

                # Check if the response contains error
                if isinstance(data, dict) and data.get("error"):
                    raise AllTubeRequestError(f"API error: {data.get('error')}")

                logger.debug("Request successful")
                video_info = VideoInfo.from_dict(data)
                logger.info(f"Successfully extracted info for: {video_info.title}")

                return video_info

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            raise AllTubeRequestError(f"Network error: {str(e)}")
        except (AllTubeAuthenticationError, AllTubeRequestError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise AllTubeError(f"Failed to extract video info: {str(e)}")
