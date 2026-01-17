"""
AllTube CDN Extractor API wrapper for extracting video information from various platforms.

This module provides a simple async interface to interact with AllTube CDN Extractor API
through RapidAPI, allowing users to extract video information, download URLs, and formats
from various video platforms (YouTube, Facebook, Instagram, TikTok, etc.).

Author: Ahmed Ayyad
"""
import json
import logging
from typing import Optional, Dict, Any

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    APIError,
    AuthenticationError,
    RequestError,
    InvalidInputError,
    APIConfig,
    with_retry,
)

try:
    from yt_dlp import YoutubeDL
    yt_dlp_installed = True
except ImportError as e:
    yt_dlp_installed = False

# Configure logging
logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

# Create aliases for backward compatibility
AllTubeError = APIError
AllTubeAuthenticationError = AuthenticationError
AllTubeRequestError = RequestError
AllTubeInvalidURLError = InvalidInputError


# ==================== AllTube API Client ====================

class AllTubeAPI(BaseRapidAPI):
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

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with AllTubeAPI(api_key="your_key") as client:
            # Get video information
            video_info = await client.get_info("https://www.youtube.com/watch?v=...")
            print(f"Title: {video_info['title']}")
            print(f"Duration: {video_info['duration']}s")

            # Get formats
            formats = video_info.get('formats', [])
            for fmt in formats:
                print(f"Format: {fmt.get('resolution')} - {fmt.get('url')}")

            # Get subtitles
            subtitles = video_info.get('subtitles', {})
            for lang, subs in subtitles.items():
                print(f"Language: {lang}")
                for sub in subs:
                    print(f"  - {sub.get('name')} ({sub.get('ext')})")

            # Use with config
            config = APIConfig(api_key="key", timeout=120)
            async with AllTubeAPI(config=config) as client:
                info = await client.get_info("url")
    """

    BASE_URL = "https://alltube-cdn-extractor.p.rapidapi.com"
    DEFAULT_HOST = "alltube-cdn-extractor.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers inherited from BaseRapidAPI

    async def get_info(self, url: str, yt_dlp_opts={}) -> Dict[str, Any]:
        """
        Extract video information from a URL.

        Args:
            url: Video URL from supported platforms (YouTube, Facebook, Instagram, etc.)

        Returns:
            Dictionary with complete video information including:
            - title: Video title
            - url: Direct video URL
            - duration: Video duration in seconds
            - formats: List of available formats with quality info
            - subtitles: Available subtitles by language
            - thumbnails: Available thumbnails
            - description: Video description
            - uploader: Channel/uploader name
            - And more...

        Raises:
            AllTubeInvalidURLError: If URL is invalid
            AllTubeAuthenticationError: If authentication fails
            AllTubeRequestError: If request fails

        Example:
            async with AllTubeAPI(api_key="key") as client:
                info = await client.get_info("https://www.youtube.com/watch?v=...")
                print(info['title'])
                print(f"Available formats: {len(info.get('formats', []))}")
        """
        if not url or not isinstance(url, str):
            raise InvalidInputError("URL must be a non-empty string")

        if not url.startswith(("http://", "https://")):
            raise InvalidInputError("URL must start with http:// or https://")

        logger.info(f"Extracting video info from: {url}")

        params = {
            "url": url,
            'yt_dlp_opts': json.dumps(yt_dlp_opts, indent=2, ensure_ascii=False)
        }

        data = await self._make_request("GET", "/getInfo", params=params)

        # Check if the response contains error
        if isinstance(data, dict) and data.get("error"):
            raise RequestError(f"API error: {data.get('error')}")

        logger.info(f"Successfully extracted info for: {data.get('title', 'Unknown')}")
        return data

    async def yt_dlp_download(self, url: str, yt_dlp_format: str = "best",
                              yt_dlp_outtmpl: str = "%(title)s.%(ext)s",download=True) -> Dict[str, Any]:
        """
        Download video using yt-dlp

        Args:
            url: Video URL
            yt_dlp_format: Format string (default: "best")
            yt_dlp_outtmpl: Output template (default: "%(title)s.%(ext)s")

        Returns:
            Dictionary containing:
            - filepath: Downloaded file path
            - info: Video information

        Raises:
            Exception: If yt-dlp is not installed
            AllTubeError: If download fails
        """
        if not yt_dlp_installed:
            raise Exception("yt_dlp is not installed. Install it with: pip install yt-dlp")

        # Build yt_dlp options
        yt_dlp_opts = {
            'format': yt_dlp_format,
            'outtmpl': yt_dlp_outtmpl
        }

        # Get video info from API
        data = await self.get_info(url, yt_dlp_opts)

        with YoutubeDL(yt_dlp_opts) as ydl:
            # Download the video
            if download:
                ydl.process_info(data)

            # Get the actual filepath
            filepath = ydl.prepare_filename(data)

            logger.info(f"Video downloaded successfully: {filepath}")

            return {
                'filepath': filepath,
                'info': data
            }