"""
AllTube CDN Extractor API wrapper for extracting video information from various platforms.

This module provides a simple async interface to interact with AllTube CDN Extractor API
through RapidAPI, allowing users to extract video information, download URLs, and formats
from various video platforms (YouTube, Facebook, Instagram, TikTok, etc.).

Author: Ahmed Ayyad
"""

import logging
from typing import Optional, Dict, Any
import aiohttp


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

    async def get_info(self, url: str) -> Dict[str, Any]:
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
                logger.info(f"Successfully extracted info for: {data.get('title', 'Unknown')}")

                return data

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            raise AllTubeRequestError(f"Network error: {str(e)}")
        except (AllTubeAuthenticationError, AllTubeRequestError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise AllTubeError(f"Failed to extract video info: {str(e)}")
