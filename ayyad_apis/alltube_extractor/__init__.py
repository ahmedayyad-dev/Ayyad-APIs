"""
AllTube CDN Extractor - Video information extraction from multiple platforms.

This module provides a clean interface to extract video information from various
platforms including YouTube, Facebook, Instagram, TikTok, Twitter, Vimeo, and more.

Example usage:
    from ayyad_apis import AllTubeAPI

    async with AllTubeAPI(api_key="your_rapidapi_key") as client:
        video_info = await client.get_info("https://www.youtube.com/watch?v=...")

        print(f"Title: {video_info['title']}")
        print(f"Duration: {video_info['duration']}s")
        print(f"Uploader: {video_info['uploader']}")

        # Get formats
        formats = video_info.get('formats', [])
        for fmt in formats:
            print(f"Resolution: {fmt.get('resolution')}")
"""

from .core import (
    # Main API client
    AllTubeAPI,

    # Exceptions
    AllTubeError,
    AllTubeAuthenticationError,
    AllTubeClientError,
    AllTubeRequestError,
    AllTubeInvalidURLError,
)

__all__ = [
    # Main API
    "AllTubeAPI",

    # Exceptions
    "AllTubeError",
    "AllTubeAuthenticationError",
    "AllTubeClientError",
    "AllTubeRequestError",
    "AllTubeInvalidURLError",
]
