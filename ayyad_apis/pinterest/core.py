"""
Pinterest API wrapper for downloading images and videos from Pinterest.

This module provides a simple async interface to interact with Pinterest API
through RapidAPI, allowing users to download images and videos from Pinterest pins.

Author: Ahmed Ayyad
"""

import logging
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import aiohttp

# Import shared download function
from ..utils import download_file


# Configure logging
logger = logging.getLogger(__name__)


# ==================== Custom Exceptions ====================

class PinterestAPIError(Exception):
    """Base exception for Pinterest API errors."""
    pass


class PinterestAuthenticationError(PinterestAPIError):
    """Raised when API authentication fails."""
    pass


class PinterestDownloadError(PinterestAPIError):
    """Raised when download operation fails."""
    pass


class PinterestInvalidURLError(PinterestAPIError):
    """Raised when Pinterest URL is invalid or malformed."""
    pass


class PinterestRequestError(PinterestAPIError):
    """Raised when API request fails."""
    pass


# ==================== Data Models ====================

@dataclass
class Thumbnail:
    """Represents a video thumbnail with dimensions."""
    url: str
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thumbnail":
        """Create Thumbnail from API response dictionary."""
        return cls(
            url=data.get("url", ""),
            width=data.get("width", 0),
            height=data.get("height", 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "width": self.width,
            "height": self.height
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact JSON)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class ImageDownloadResult:
    """Result from image download operation."""
    success: bool
    download_url: str
    title: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "download_url": self.download_url,
            "title": self.title
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact JSON)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        return_bytes: bool = False,
        show_progress: bool = False,
        max_retries: int = 3
    ) -> Union[bytes, str, None]:
        """
        Download the actual image file - shortcut to download_file().

        Args:
            output_path: Path to save the file. If None, generates from title.
            return_bytes: If True, returns bytes instead of saving to file.
            show_progress: Show download progress in console (default: False)
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            - bytes if return_bytes=True
            - str (file path) if saved to disk
            - None if download fails

        Example:
            result = await client.image("https://pinterest.com/pin/...")
            await result.download("my_image.jpg")
        """
        if not self.download_url:
            logger.error("No download URL available")
            return None

        # Generate filename from title if no output_path provided
        if output_path is None and not return_bytes:
            safe_title = "".join(c for c in self.title if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title.strip()[:50] or 'pinterest_image'
            output_path = safe_title

        # Use the general download_file function
        return await download_file(
            url=self.download_url,
            output_path=output_path,
            return_bytes=return_bytes,
            default_filename="pinterest_image",
            default_ext=".jpg",
            show_progress=show_progress,
            max_retries=max_retries
        )


@dataclass
class VideoDownloadResult:
    """Result from video download operation."""
    success: bool
    download_url: str
    title: str
    thumbnails: List[Thumbnail]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "download_url": self.download_url,
            "title": self.title,
            "thumbnails": [
                {"url": t.url, "width": t.width, "height": t.height}
                for t in self.thumbnails
            ]
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact JSON)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_thumbnail_by_size(self, min_width: int = 0, min_height: int = 0) -> Optional[Thumbnail]:
        """
        Get a thumbnail matching minimum dimensions.

        Args:
            min_width: Minimum width required
            min_height: Minimum height required

        Returns:
            First thumbnail matching criteria or None
        """
        for thumb in self.thumbnails:
            if thumb.width >= min_width and thumb.height >= min_height:
                return thumb
        return None

    def get_largest_thumbnail(self) -> Optional[Thumbnail]:
        """Get the largest thumbnail available."""
        if not self.thumbnails:
            return None
        return max(self.thumbnails, key=lambda t: t.width * t.height)

    async def download(
        self,
        output_path: Optional[Union[str, Path]] = None,
        return_bytes: bool = False,
        download_thumbnails: bool = False,
        thumbnails_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        max_retries: int = 3
    ) -> Union[bytes, str, Dict[str, Any], None]:
        """
        Download the actual video file - shortcut to download_file().

        Args:
            output_path: Path to save the video. If None, generates from title.
            return_bytes: If True, returns bytes instead of saving to file.
            download_thumbnails: If True, also downloads all thumbnails.
            thumbnails_dir: Directory to save thumbnails (default: same as video).
            show_progress: Show download progress in console (default: True)
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            - bytes if return_bytes=True
            - str (file path) if saved to disk without thumbnails
            - dict with video path and thumbnail paths if download_thumbnails=True
            - None if download fails

        Example:
            result = await client.video("https://pinterest.com/pin/...")
            paths = await result.download(download_thumbnails=True)
        """
        if not self.download_url:
            logger.error("No download URL available")
            return None

        # Generate filename from title if no output_path provided
        if output_path is None and not return_bytes:
            safe_title = "".join(c for c in self.title if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title.strip()[:50] or 'pinterest_video'
            output_path = safe_title

        # Download main video using general function
        video_path = await download_file(
            url=self.download_url,
            output_path=output_path,
            return_bytes=return_bytes,
            default_filename="pinterest_video",
            default_ext=".mp4",
            show_progress=show_progress,
            max_retries=max_retries
        )

        if video_path is None:
            return None

        # Return bytes directly if requested
        if return_bytes:
            return video_path  # This is actually bytes

        # Download thumbnails if requested
        if download_thumbnails and self.thumbnails:
            thumb_dir = Path(thumbnails_dir) if thumbnails_dir else Path(video_path).parent
            thumb_dir.mkdir(parents=True, exist_ok=True)

            thumbnail_paths = []
            base_name = Path(video_path).stem

            for idx, thumb in enumerate(self.thumbnails):
                thumb_filename = f"{base_name}_thumb_{idx}_{thumb.width}x{thumb.height}"
                thumb_path = await download_file(
                    url=thumb.url,
                    output_path=str(thumb_dir / thumb_filename),
                    default_ext=".jpg"
                )

                if thumb_path:
                    thumbnail_paths.append(thumb_path)
                    logger.debug(f"Thumbnail saved: {thumb_path}")

            return {
                "video": video_path,
                "thumbnails": thumbnail_paths
            }

        return video_path


# ==================== Pinterest API Client ====================

class PinterestAPI:
    """
    Async client for Pinterest API via RapidAPI.

    Example:
        async with PinterestAPI(api_key="your_key") as client:
            # Get image metadata
            img_result = await client.image("https://www.pinterest.com/pin/...")
            print(img_result.download_url)

            # Download image file
            await img_result.download("my_image.jpg")

            # Get video metadata + thumbnails
            vid_result = await client.video("https://www.pinterest.com/pin/...")

            # Download video with thumbnails
            paths = await vid_result.download(download_thumbnails=True)
            print(paths)  # {"video": "...", "thumbnails": [...]}
    """

    BASE_URL = "https://pinterest-api4.p.rapidapi.com"
    DEFAULT_HOST = "pinterest-api4.p.rapidapi.com"

    def __init__(
        self,
        api_key: str,
        rapidapi_host: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Pinterest API client.

        Args:
            api_key: RapidAPI key for authentication
            rapidapi_host: RapidAPI host (defaults to pinterest-api4.p.rapidapi.com)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.rapidapi_host = rapidapi_host or self.DEFAULT_HOST
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info("Pinterest API client initialized")

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

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make an async GET request to the API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            PinterestAuthenticationError: If authentication fails
            PinterestRequestError: If request fails
        """
        if not self._session:
            raise PinterestAPIError("Session not initialized. Use async context manager.")

        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers()

        logger.debug(f"Making request to {endpoint} with params {params}")

        try:
            async with self._session.get(url, headers=headers, params=params) as response:
                # Check for authentication errors
                if response.status == 401 or response.status == 403:
                    raise PinterestAuthenticationError(
                        f"Authentication failed: {response.status}"
                    )

                # Check for other errors
                if response.status != 200:
                    error_text = await response.text()
                    raise PinterestRequestError(
                        f"Request failed with status {response.status}: {error_text}"
                    )

                data = await response.json()
                logger.debug(f"Request successful: {endpoint}")
                return data

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            raise PinterestRequestError(f"Network error: {str(e)}")

    async def image(self, url: str) -> ImageDownloadResult:
        """
        Download an image from Pinterest.

        Args:
            url: Pinterest pin URL (e.g., https://www.pinterest.com/pin/12345/)

        Returns:
            ImageDownloadResult with download URL and title

        Raises:
            PinterestInvalidURLError: If URL is invalid
            PinterestDownloadError: If download fails
        """
        if not url or not isinstance(url, str):
            raise PinterestInvalidURLError("URL must be a non-empty string")

        if "pinterest.com" not in url:
            raise PinterestInvalidURLError("URL must be a valid Pinterest URL")

        logger.info(f"Downloading image from: {url}")

        try:
            params = {"url": url}
            data = await self._make_request("/download/image", params)

            if not data.get("success"):
                raise PinterestDownloadError("API returned success=false")

            result = ImageDownloadResult(
                success=data.get("success", False),
                download_url=data.get("download_url", ""),
                title=data.get("title", "")
            )

            logger.info(f"Image downloaded successfully: {result.title}")
            return result

        except (PinterestAuthenticationError, PinterestRequestError):
            raise
        except Exception as e:
            logger.error(f"Image download failed: {str(e)}")
            raise PinterestDownloadError(f"Failed to download image: {str(e)}")

    async def video(self, url: str) -> VideoDownloadResult:
        """
        Download a video from Pinterest.

        Args:
            url: Pinterest pin URL (e.g., https://www.pinterest.com/pin/12345/)

        Returns:
            VideoDownloadResult with download URL, title, and thumbnails

        Raises:
            PinterestInvalidURLError: If URL is invalid
            PinterestDownloadError: If download fails
        """
        if not url or not isinstance(url, str):
            raise PinterestInvalidURLError("URL must be a non-empty string")

        if "pinterest.com" not in url:
            raise PinterestInvalidURLError("URL must be a valid Pinterest URL")

        logger.info(f"Downloading video from: {url}")

        try:
            params = {"url": url}
            data = await self._make_request("/download/video", params)

            if not data.get("success"):
                raise PinterestDownloadError("API returned success=false")

            # Parse thumbnails
            thumbnails = []
            for thumb_data in data.get("thumbnails", []):
                thumbnails.append(Thumbnail.from_dict(thumb_data))

            result = VideoDownloadResult(
                success=data.get("success", False),
                download_url=data.get("download_url", ""),
                title=data.get("title", ""),
                thumbnails=thumbnails
            )

            logger.info(f"Video downloaded successfully: {result.title}")
            return result

        except (PinterestAuthenticationError, PinterestRequestError):
            raise
        except Exception as e:
            logger.error(f"Video download failed: {str(e)}")
            raise PinterestDownloadError(f"Failed to download video: {str(e)}")

    async def download(self, url: str, media_type: str = "auto") -> Union[ImageDownloadResult, VideoDownloadResult]:
        """
        Download media from Pinterest (auto-detect or specify type).

        Args:
            url: Pinterest pin URL
            media_type: "image", "video", or "auto" to try both

        Returns:
            ImageDownloadResult or VideoDownloadResult

        Raises:
            PinterestInvalidURLError: If URL or media_type is invalid
            PinterestDownloadError: If download fails

        Example:
            async with PinterestAPI(api_key="key") as client:
                # Auto-detect type
                result = await client.download("https://pinterest.com/pin/...")

                # Force image
                result = await client.download(url, media_type="image")
        """
        if media_type not in ["image", "video", "auto"]:
            raise PinterestInvalidURLError(
                "media_type must be 'image', 'video', or 'auto'"
            )

        if media_type == "image":
            return await self.image(url)
        elif media_type == "video":
            return await self.video(url)
        else:  # auto
            # Try image first, then video
            try:
                return await self.image(url)
            except PinterestDownloadError:
                logger.debug("Image download failed, trying video...")
                return await self.video(url)

    # Aliases for backward compatibility
    async def download_image(self, url: str) -> ImageDownloadResult:
        """Alias for image(). Download an image from Pinterest."""
        return await self.image(url)

    async def download_video(self, url: str) -> VideoDownloadResult:
        """Alias for video(). Download a video from Pinterest."""
        return await self.video(url)
