"""
Shared utilities for Ayyad APIs library.

This module provides common utility functions used across different API modules.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import aiohttp
import aiofiles


# Configure logging
logger = logging.getLogger(__name__)


async def download_file(
    url: str,
    output_path: Optional[Union[str, Path]] = None,
    return_bytes: bool = False,
    default_filename: str = "download",
    default_ext: str = ".bin"
) -> Union[bytes, str, None]:
    """
    Download a file from URL - unified function for the entire library.

    This is a shared utility function used by all API modules to download
    media files (images, videos, audio) from URLs.

    Args:
        url: URL to download from
        output_path: Path to save the file. If None, uses default_filename + extension from URL
        return_bytes: If True, returns bytes instead of saving to file
        default_filename: Default filename if can't determine from URL
        default_ext: Default extension if can't determine from URL

    Returns:
        - bytes if return_bytes=True
        - str (file path) if saved to disk
        - None if download fails

    Example:
        # Download to specific path
        path = await download_file("https://example.com/video.mp4", "my_video.mp4")

        # Get as bytes
        data = await download_file("https://example.com/image.jpg", return_bytes=True)

        # Auto-generate filename from URL
        path = await download_file("https://example.com/image.jpg")

    Note:
        This function is used internally by all media result classes (VideoInfo,
        Format, ImageDownloadResult, VideoDownloadResult, etc.) in their
        download() methods.
    """
    if not url:
        logger.error("No URL provided")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download: HTTP {response.status}")
                    return None

                content = await response.read()

                if return_bytes:
                    return content

                # Determine output path
                if output_path is None:
                    # Try to extract extension from URL
                    ext = default_ext
                    if "." in url:
                        url_ext = url.split(".")[-1].split("?")[0]
                        if url_ext and len(url_ext) <= 5:  # Reasonable extension length
                            ext = f".{url_ext}"

                    output_path = f"{default_filename}{ext}"

                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                async with aiofiles.open(output_path, "wb") as f:
                    await f.write(content)

                logger.info(f"File saved to: {output_path}")
                return str(output_path)

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return None
