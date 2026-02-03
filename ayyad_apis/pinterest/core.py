"""
Pinterest API wrapper for downloading images and videos from Pinterest.

This module provides a simple async interface to interact with Pinterest API
through RapidAPI, allowing users to download images and videos from Pinterest pins.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# Import shared utilities and base classes
from ..utils import (
    download_file,
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    RequestError,
    InvalidInputError,
    DownloadError,
)

# Configure logging
logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

# Create aliases for backward compatibility
PinterestAPIError = APIError
PinterestAuthenticationError = AuthenticationError
PinterestDownloadError = DownloadError
PinterestInvalidURLError = InvalidInputError
PinterestRequestError = RequestError


# ==================== Data Models ====================

@dataclass
class Thumbnail(BaseResponse):
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

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class ImageMetadata(BaseResponse):
    """Image metadata (dimensions, size, format)."""
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    format: Optional[str] = None

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class ImageDownloadResult(BaseResponse):
    """Result from image download operation."""
    success: bool
    download_url: str
    title: str
    metadata: Optional[ImageMetadata] = None
    cached: bool = False

    # to_dict() and to_json() inherited from BaseResponse

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
class VideoDownloadResult(BaseResponse):
    """Result from video download operation."""
    success: bool
    download_url: str
    title: str
    thumbnails: List[Thumbnail]

    # to_dict() and to_json() inherited from BaseResponse

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


@dataclass
class BoardDownloadResult(BaseResponse):
    """Result from board download operation."""
    success: bool
    board_name: str
    total_images: int
    images: List[Dict[str, Any]]
    requests_used: int = 1
    zip_url: Optional[str] = None

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class BatchDownloadResult(BaseResponse):
    """Result from batch download operation."""
    success: bool
    total_requested: int
    total_downloaded: int
    failed: int
    images: List[Dict[str, Any]]
    requests_used: int = 1
    zip_url: Optional[str] = None

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class ProfileDownloadResult(BaseResponse):
    """Result from profile download operation."""
    success: bool
    username: str
    total_pins: int
    images: List[Dict[str, Any]]
    requests_used: int = 1
    zip_url: Optional[str] = None

    # to_dict() and to_json() inherited from BaseResponse


# ==================== Pinterest API Client ====================

class PinterestAPI(BaseRapidAPI):
    """
    Async client for Pinterest API via RapidAPI.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

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

            # Use with config
            config = APIConfig(api_key="key", max_retries=5)
            async with PinterestAPI(config=config) as client:
                result = await client.image("url")
    """

    BASE_URL = "https://pinterest-api4.p.rapidapi.com"
    DEFAULT_HOST = "pinterest-api4.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers, _make_request inherited from BaseRapidAPI

    async def image(
        self,
        url: str,
        quality: str = "original",
        format: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> ImageDownloadResult:
        """
        Download an image from Pinterest.

        Args:
            url: Pinterest pin URL (e.g., https://www.pinterest.com/pin/12345/)
            quality: Image quality - 'original', 'high', 'medium', 'low' (default: 'original')
            format: Output format - 'jpeg', 'png', 'webp' (optional)
            width: Resize width (maintains aspect ratio, optional)
            height: Resize height (maintains aspect ratio, optional)

        Returns:
            ImageDownloadResult with download URL, title, and metadata

        Raises:
            PinterestInvalidURLError: If URL is invalid
            PinterestDownloadError: If download fails
        """
        if not url or not isinstance(url, str):
            raise InvalidInputError("URL must be a non-empty string")

        if "pinterest.com" not in url and "pin.it" not in url:
            raise InvalidInputError("URL must be a valid Pinterest URL")

        logger.info(f"Downloading image from: {url} (quality={quality}, format={format}, size={width}x{height})")

        params = {"url": url, "quality": quality}
        if format:
            params["format"] = format
        if width:
            params["width"] = str(width)
        if height:
            params["height"] = str(height)

        data = await self._make_request("GET", "/download/image", params=params)

        if not data.get("success"):
            raise DownloadError("API returned success=false")

        # Parse metadata if available
        metadata = None
        if "metadata" in data:
            meta_data = data["metadata"]
            metadata = ImageMetadata(
                width=meta_data.get("width"),
                height=meta_data.get("height"),
                file_size=meta_data.get("file_size"),
                format=meta_data.get("format")
            )

        result = ImageDownloadResult(
            success=data.get("success", False),
            download_url=data.get("download_url", ""),
            title=data.get("title", ""),
            metadata=metadata,
            cached=data.get("cached", False)
        )

        logger.info(f"Image downloaded successfully: {result.title}")
        return result

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
            raise InvalidInputError("URL must be a non-empty string")

        if "pinterest.com" not in url:
            raise InvalidInputError("URL must be a valid Pinterest URL")

        logger.info(f"Downloading video from: {url}")

        params = {"url": url}
        data = await self._make_request("GET", "/download/video", params=params)

        if not data.get("success"):
            raise DownloadError("API returned success=false")

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
            raise InvalidInputError(
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

    async def board(
        self,
        url: str,
        max_images: int = 25,
        return_zip: bool = False
    ) -> BoardDownloadResult:
        """
        Download images from a Pinterest board.

        Args:
            url: Pinterest board URL
            max_images: Maximum number of images to download (1-50, default: 25)
            return_zip: Return as ZIP file (default: False)

        Returns:
            BoardDownloadResult with board info and image URLs

        Raises:
            PinterestInvalidURLError: If URL is invalid
            PinterestDownloadError: If download fails
        """
        if not url or not isinstance(url, str):
            raise InvalidInputError("URL must be a non-empty string")

        if "pinterest.com" not in url and "pin.it" not in url:
            raise InvalidInputError("URL must be a valid Pinterest URL")

        logger.info(f"Downloading board: {url} (max_images={max_images}, return_zip={return_zip})")

        params = {
            "url": url,
            "max_images": str(max_images),
            "return_zip": str(return_zip).lower()
        }
        data = await self._make_request("GET", "/download/board", params=params)

        if not data.get("success"):
            raise DownloadError("API returned success=false")

        result = BoardDownloadResult(
            success=data.get("success", False),
            board_name=data.get("board_name", ""),
            total_images=data.get("total_images", 0),
            images=data.get("images", []),
            requests_used=data.get("requests_used", 1),
            zip_url=data.get("zip_url")
        )

        logger.info(f"Board downloaded successfully: {result.board_name} ({result.total_images} images)")
        return result

    async def batch(
        self,
        urls: List[str],
        quality: str = "original",
        return_zip: bool = False
    ) -> BatchDownloadResult:
        """
        Download multiple images in batch.

        Args:
            urls: List of Pinterest URLs (max 50)
            quality: Image quality - 'original', 'high', 'medium', 'low' (default: 'original')
            return_zip: Return as ZIP file (default: False)

        Returns:
            BatchDownloadResult with download info and image URLs

        Raises:
            PinterestInvalidURLError: If URLs are invalid or exceed limit
            PinterestDownloadError: If download fails
        """
        if not urls or not isinstance(urls, list):
            raise InvalidInputError("URLs must be a non-empty list")

        if len(urls) > 50:
            raise InvalidInputError("Maximum 50 URLs per batch")

        logger.info(f"Batch download: {len(urls)} URLs (quality={quality}, return_zip={return_zip})")

        payload = {
            "urls": urls,
            "quality": quality,
            "return_zip": return_zip
        }

        data = await self._make_request("POST", "/download/batch", json=payload)

        if not data.get("success"):
            raise DownloadError("API returned success=false")

        result = BatchDownloadResult(
            success=data.get("success", False),
            total_requested=data.get("total_requested", 0),
            total_downloaded=data.get("total_downloaded", 0),
            failed=data.get("failed", 0),
            images=data.get("images", []),
            requests_used=data.get("requests_used", 1),
            zip_url=data.get("zip_url")
        )

        logger.info(f"Batch download completed: {result.total_downloaded}/{result.total_requested} successful")
        return result

    async def profile(
        self,
        url: str,
        max_pins: int = 30,
        return_zip: bool = False
    ) -> ProfileDownloadResult:
        """
        Download pins from a Pinterest profile.

        Args:
            url: Pinterest profile URL
            max_pins: Maximum number of pins to download (1-50, default: 30)
            return_zip: Return as ZIP file (default: False)

        Returns:
            ProfileDownloadResult with profile info and image URLs

        Raises:
            PinterestInvalidURLError: If URL is invalid
            PinterestDownloadError: If download fails
        """
        if not url or not isinstance(url, str):
            raise InvalidInputError("URL must be a non-empty string")

        if "pinterest.com" not in url and "pin.it" not in url:
            raise InvalidInputError("URL must be a valid Pinterest URL")

        logger.info(f"Downloading profile: {url} (max_pins={max_pins}, return_zip={return_zip})")

        params = {
            "url": url,
            "max_pins": str(max_pins),
            "return_zip": str(return_zip).lower()
        }
        data = await self._make_request("GET", "/download/profile", params=params)

        if not data.get("success"):
            raise DownloadError("API returned success=false")

        result = ProfileDownloadResult(
            success=data.get("success", False),
            username=data.get("username", ""),
            total_pins=data.get("total_pins", 0),
            images=data.get("images", []),
            requests_used=data.get("requests_used", 1),
            zip_url=data.get("zip_url")
        )

        logger.info(f"Profile downloaded successfully: {result.username} ({result.total_pins} pins)")
        return result

    # Aliases for backward compatibility
    async def download_image(self, url: str) -> ImageDownloadResult:
        """Alias for image(). Download an image from Pinterest."""
        return await self.image(url)

    async def download_video(self, url: str) -> VideoDownloadResult:
        """Alias for video(). Download a video from Pinterest."""
        return await self.video(url)
