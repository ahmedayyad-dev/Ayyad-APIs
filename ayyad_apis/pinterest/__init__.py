"""
Pinterest API module for downloading images and videos.

This module provides async access to Pinterest content through RapidAPI.
"""

# Import shared utility function
from ..utils import download_file

from .core import (
    # Main API Client
    PinterestAPI,

    # Data Models
    Thumbnail,
    ImageDownloadResult,
    VideoDownloadResult,

    # Exceptions
    PinterestAPIError,
    PinterestAuthenticationError,
    PinterestDownloadError,
    PinterestInvalidURLError,
    PinterestRequestError,
)

__all__ = [
    # Utility Functions
    "download_file",

    # Main API Client
    "PinterestAPI",

    # Data Models
    "Thumbnail",
    "ImageDownloadResult",
    "VideoDownloadResult",

    # Exceptions
    "PinterestAPIError",
    "PinterestAuthenticationError",
    "PinterestDownloadError",
    "PinterestInvalidURLError",
    "PinterestRequestError",
]
