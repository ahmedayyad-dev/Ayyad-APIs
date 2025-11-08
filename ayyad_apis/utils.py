"""
Shared utilities for Ayyad APIs library.

This module provides common utility functions used across different API modules.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional, Union, Callable
import aiohttp
import aiofiles


# Configure logging
logger = logging.getLogger(__name__)


async def download_file(
    url: str,
    output_path: Optional[Union[str, Path]] = None,
    return_bytes: bool = False,
    default_filename: str = "download",
    default_ext: str = ".bin",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 8192,
    session: Optional[aiohttp.ClientSession] = None
) -> Union[bytes, str, None]:
    """
    Download a file from URL - unified function for the entire library.

    This is a shared utility function used by all API modules to download
    media files (images, videos, audio) from URLs with support for:
    - Progress tracking
    - Retry logic
    - Custom chunk sizes
    - Reusable sessions

    Args:
        url: URL to download from
        output_path: Path to save the file. If None, uses default_filename + extension from URL
        return_bytes: If True, returns bytes instead of saving to file
        default_filename: Default filename if can't determine from URL
        default_ext: Default extension if can't determine from URL
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay in seconds between retries (default: 2.0)
        show_progress: Show download progress in console (default: False)
        progress_callback: Optional callback function(downloaded_bytes, total_bytes)
        chunk_size: Size of chunks for streaming download (default: 8192)
        session: Optional aiohttp session to reuse (if None, creates new one)

    Returns:
        - bytes if return_bytes=True
        - str (file path) if saved to disk
        - None if download fails

    Example:
        # Simple download
        path = await download_file("https://example.com/video.mp4", "my_video.mp4")

        # Download with progress
        path = await download_file(
            "https://example.com/large_video.mp4",
            "video.mp4",
            show_progress=True,
            max_retries=5
        )

        # Get as bytes
        data = await download_file("https://example.com/image.jpg", return_bytes=True)

        # Custom progress callback
        def on_progress(downloaded, total):
            print(f"Downloaded: {downloaded}/{total} bytes")

        path = await download_file(
            "https://example.com/file.zip",
            progress_callback=on_progress
        )

    Note:
        This function is used internally by all media result classes (VideoInfo,
        Format, ImageDownloadResult, VideoDownloadResult, etc.) in their
        download() methods.
    """
    if not url:
        logger.error("No URL provided")
        return None

    # Determine output path for file downloads
    final_output_path = None
    if not return_bytes:
        if output_path is None:
            # Try to extract extension from URL
            ext = default_ext
            if "." in url:
                url_ext = url.split(".")[-1].split("?")[0]
                if url_ext and len(url_ext) <= 5:  # Reasonable extension length
                    ext = f".{url_ext}"
            output_path = f"{default_filename}{ext}"

        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Retry logic
    last_error = None
    close_session = False

    for attempt in range(max_retries):
        try:
            # Create session if not provided
            if session is None:
                session = aiohttp.ClientSession()
                close_session = True

            if attempt > 0:
                logger.info(f"[Download] Retry attempt {attempt + 1}/{max_retries} for: {url}")

            async with session.get(url) as response:
                if response.status != 200:
                    error_msg = f"HTTP {response.status}: Download failed"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                total_size = int(response.headers.get('content-length', 0))

                if return_bytes:
                    # Simple read for bytes
                    if show_progress and total_size > 0:
                        logger.info(f"[Download] Downloading {total_size:,} bytes from: {url}")

                    content = await response.read()

                    if show_progress:
                        logger.info(f"[Download] Completed: {len(content):,} bytes")

                    # Close session if we created it
                    if close_session:
                        await session.close()
                        session = None

                    return content

                # Streaming download to file
                if show_progress:
                    if total_size > 0:
                        logger.info(f"[Download] Starting download: {final_output_path} ({total_size:,} bytes)")
                    else:
                        logger.info(f"[Download] Starting download: {final_output_path}")

                downloaded = 0
                last_logged_percent = 0

                async with aiofiles.open(final_output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)

                        # Progress tracking
                        if total_size > 0:
                            percentage = (downloaded / total_size) * 100

                            # Console progress
                            if show_progress:
                                print(f"\r[Download] Progress: {percentage:.1f}% ({downloaded:,}/{total_size:,} bytes)", end="", flush=True)

                                # Log every 10%
                                current_milestone = int(percentage // 10) * 10
                                if current_milestone > last_logged_percent and current_milestone > 0:
                                    print()  # New line
                                    logger.info(f"[Download] {current_milestone}% completed")
                                    last_logged_percent = current_milestone

                            # Custom callback
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                        else:
                            # Unknown size
                            if show_progress:
                                print(f"\r[Download] Downloaded: {downloaded:,} bytes", end="", flush=True)

                            if progress_callback:
                                progress_callback(downloaded, 0)

                if show_progress:
                    print()  # New line after progress
                    logger.info(f"[Download] Completed: {final_output_path} ({downloaded:,} bytes)")

                # Close session if we created it
                if close_session:
                    await session.close()
                    session = None

                return str(final_output_path)

        except Exception as e:
            last_error = e
            logger.error(f"[Download] Attempt {attempt + 1} failed: {str(e)}")

            # Close session on error if we created it
            if close_session and session:
                await session.close()
                session = None

            # Retry if not last attempt
            if attempt < max_retries - 1:
                logger.warning(f"[Download] Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"[Download] All {max_retries} attempts failed")

    # All retries failed
    logger.error(f"[Download] Failed to download from: {url}")
    return None
