"""
Porn Detection API wrapper for detecting NSFW content in images and videos.

This module provides a simple async interface to interact with Porn Detection API
through RapidAPI, allowing users to detect pornographic content in images and videos.

Author: Ahmed Ayyad
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import aiofiles
import aiohttp

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    ClientError,
    RequestError,
    InvalidInputError,
    APIConfig,
    with_retry,
)

logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

DetectionError = APIError
APIResponseError = RequestError
UploadError = RequestError


# ==================== Video Analysis Configuration ====================

@dataclass
class VideoAnalysisConfig(BaseResponse):
    """Configuration for video analysis"""
    start_sec: float = 0.0
    duration_sec: float = 0.0  # 0.0 = analyze the entire video
    thresh_high: float = 0.70
    thresh_low: float = 0.60
    min_hit_duration: float = 1.0
    min_ratio: float = 0.02
    smooth_window: int = 5

    def to_params(self) -> Dict[str, str]:
        """Convert settings into API parameters"""
        return {
            "start_sec": str(self.start_sec),
            "duration_sec": str(self.duration_sec),
            "thresh_high": str(self.thresh_high),
            "thresh_low": str(self.thresh_low),
            "min_hit_duration": str(self.min_hit_duration),
            "min_ratio": str(self.min_ratio),
            "smooth_window": str(self.smooth_window)
        }

    # to_dict() and to_json() inherited from BaseResponse


# ==================== Image Detection Result ====================

@dataclass
class ImageDetectionResult(BaseResponse):
    """Result of image content detection"""
    label: str = "Safe"  # "Unsafe" or "Safe"
    nsfw_prob: float = 0.0
    threshold: float = 0.7
    success: bool = True

    @property
    def is_nsfw(self) -> bool:
        """Check if content is unsafe"""
        return self.label == "Unsafe"

    @property
    def is_safe(self) -> bool:
        """Check if content is safe"""
        return self.label == "Safe"

    @property
    def confidence_percentage(self) -> str:
        """Confidence level as percentage"""
        return f"{self.nsfw_prob * 100:.1f}%"

    @property
    def safety_level(self) -> str:
        """Safety level description"""
        if self.is_safe:
            return "Safe"
        elif self.nsfw_prob >= 0.9:
            return "High Risk"
        elif self.nsfw_prob >= 0.7:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties."""
        return {
            "label": self.label,
            "nsfw_prob": self.nsfw_prob,
            "threshold": self.threshold,
            "success": self.success,
            "is_nsfw": self.is_nsfw,
            "confidence_percentage": self.confidence_percentage,
            "safety_level": self.safety_level
        }

    # to_json() inherited from BaseResponse


# ==================== Video Thresholds ====================

@dataclass
class VideoThresholds(BaseResponse):
    """Thresholds used in video analysis"""
    thresh_high: float = 0.8
    thresh_low: float = 0.7
    min_hit_duration: float = 1.0
    min_ratio: float = 0.02

    # to_dict() and to_json() inherited from BaseResponse


# ==================== Video Statistics ====================

@dataclass
class VideoStats(BaseResponse):
    """Detailed video analysis statistics"""
    max_prob: float = 0.0
    avg_prob: float = 0.0
    total_duration: float = 0.0
    total_above_duration: float = 0.0
    ratio_above: float = 0.0
    max_streak: float = 0.0
    sample_step: float = 0.0
    num_samples: int = 0

    @property
    def max_prob_percentage(self) -> str:
        """Maximum probability as percentage"""
        return f"{self.max_prob * 100:.1f}%"

    @property
    def avg_prob_percentage(self) -> str:
        """Average probability as percentage"""
        return f"{self.avg_prob * 100:.1f}%"

    @property
    def ratio_above_percentage(self) -> str:
        """Ratio above threshold as percentage"""
        return f"{self.ratio_above * 100:.1f}%"

    @property
    def total_duration_formatted(self) -> str:
        """Formatted total video duration"""
        return self._format_duration(self.total_duration)

    @property
    def total_above_duration_formatted(self) -> str:
        """Formatted unsafe content duration"""
        return self._format_duration(self.total_above_duration)

    @property
    def max_streak_formatted(self) -> str:
        """Formatted longest continuous unsafe streak"""
        return self._format_duration(self.max_streak)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS"""
        hours: int = int(seconds // 3600)
        minutes: int = int((seconds % 3600) // 60)
        secs: int = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties."""
        data: Dict[str, Any] = asdict(self)
        data.update({
            "max_prob_percentage": self.max_prob_percentage,
            "avg_prob_percentage": self.avg_prob_percentage,
            "ratio_above_percentage": self.ratio_above_percentage,
            "total_duration_formatted": self.total_duration_formatted,
            "total_above_duration_formatted": self.total_above_duration_formatted,
            "max_streak_formatted": self.max_streak_formatted
        })
        return data

    # to_json() inherited from BaseResponse


# ==================== Video Detection Result ====================

@dataclass
class VideoDetectionResult(BaseResponse):
    """Result of video content detection"""
    nsfw: bool = False
    reason: str = ""
    thresholds: Optional[VideoThresholds] = None
    stats: Optional[VideoStats] = None
    success: bool = True

    @property
    def is_nsfw(self) -> bool:
        """Check if content is unsafe"""
        return self.nsfw

    @property
    def is_safe(self) -> bool:
        """Check if content is safe"""
        return not self.nsfw

    @property
    def safety_level(self) -> str:
        """Determine safety level based on statistics"""
        if self.is_safe:
            return "Safe"
        elif self.stats and self.stats.max_prob >= 0.9:
            return "High Risk"
        elif self.stats and self.stats.max_prob >= 0.7:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties."""
        return {
            "nsfw": self.nsfw,
            "reason": self.reason,
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "stats": self.stats.to_dict() if self.stats else None,
            "success": self.success,
            "is_nsfw": self.is_nsfw,
            "safety_level": self.safety_level
        }

    # to_json() inherited from BaseResponse


# ==================== Upload URL Information ====================

@dataclass
class UploadUrl(BaseResponse):
    """Video upload URL"""
    url: str
    key: str = ""

    def __post_init__(self) -> None:
        """Extract key from the URL"""
        if "key=" in self.url:
            self.key = self.url.split("key=")[1].split("&")[0]

    # to_dict() and to_json() inherited from BaseResponse


# ==================== API Client ====================

class PornDetectionAPI(BaseRapidAPI):
    """
    API wrapper for pornographic content detection.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with PornDetectionAPI(api_key="key") as client:
            result = await client.predict_image_url("https://example.com/image.jpg")
            print(result.label)

            # Use with config
            config = APIConfig(api_key="key", timeout=60, max_retries=5)
            async with PornDetectionAPI(config=config) as client:
                result = await client.predict_image_url("url")
    """

    BASE_URL = "https://porn-detection-api.p.rapidapi.com"
    DEFAULT_HOST = "porn-detection-api.p.rapidapi.com"

    def __init__(self, api_key: str, timeout: int = 60, max_retries: int = 3, retry_delay: float = 1.0,
                 config: Optional[APIConfig] = None) -> None:
        super().__init__(api_key=api_key, timeout=timeout, config=config)
        self._max_retries: int = max_retries
        self._retry_delay: float = retry_delay

    # __aenter__ and __aexit__ inherited from BaseRapidAPI

    # -------------------- Response Parsers --------------------

    def _parse_image_response(self, data: Dict[str, Any]) -> ImageDetectionResult:
        """Parse image detection response"""
        try:
            return ImageDetectionResult(
                label=data.get("label", "Safe"),
                nsfw_prob=data.get("nsfw_prob", 0.0),
                threshold=data.get("threshold", 0.7),
                success=True
            )
        except Exception as e:
            logger.warning(f"Failed to parse image response: {e}")
            return ImageDetectionResult(success=False)

    def _parse_video_response(self, data: Dict[str, Any]) -> VideoDetectionResult:
        """Parse video detection response"""
        try:
            thresholds_data: Dict[str, Any] = data.get("thresholds", {})
            thresholds = VideoThresholds(
                thresh_high=thresholds_data.get("thresh_high", 0.8),
                thresh_low=thresholds_data.get("thresh_low", 0.7),
                min_hit_duration=thresholds_data.get("min_hit_duration", 1.0),
                min_ratio=thresholds_data.get("min_ratio", 0.02)
            )

            stats_data: Dict[str, Any] = data.get("stats", {})
            stats = VideoStats(
                max_prob=stats_data.get("max_prob", 0.0),
                avg_prob=stats_data.get("avg_prob", 0.0),
                total_duration=stats_data.get("total_duration", 0.0),
                total_above_duration=stats_data.get("total_above_duration", 0.0),
                ratio_above=stats_data.get("ratio_above", 0.0),
                max_streak=stats_data.get("max_streak", 0.0),
                sample_step=stats_data.get("sample_step", 0.0),
                num_samples=stats_data.get("num_samples", 0)
            )

            return VideoDetectionResult(
                nsfw=data.get("nsfw", False),
                reason=data.get("reason", ""),
                thresholds=thresholds,
                stats=stats,
                success=True
            )
        except Exception as e:
            logger.warning(f"Failed to parse video response: {e}")
            return VideoDetectionResult(success=False)

    def _parse_upload_url_response(self, data: Dict[str, Any]) -> UploadUrl:
        """Parse response for upload URL request"""
        try:
            url: str = data.get("url", "")
            return UploadUrl(url=url)
        except Exception as e:
            logger.error(f"Failed to parse upload URL response: {e}")
            raise APIResponseError(f"Invalid upload URL response: {e}")

    # -------------------- File Upload Helper --------------------

    async def _make_file_request(
        self,
        url: str,
        file_path: str,
        params: Optional[Dict[str, str]] = None,
        is_external: bool = False
    ) -> Dict[str, Any]:
        """
        Send a POST request with file upload.

        Args:
            url: Full URL or endpoint path (if not external)
            file_path: Path to the file to upload
            params: Optional query parameters
            is_external: If True, url is treated as a full URL; otherwise as an endpoint

        Returns:
            JSON response as dictionary

        Raises:
            InvalidInputError: If file not found
            AuthenticationError: If authentication fails
            UploadError/RequestError: If upload fails
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        full_url: str = url if is_external else f"{self.BASE_URL}/{url}"

        if not Path(file_path).exists():
            raise InvalidInputError(f"File not found: {file_path}")

        logger.info(f"POST {full_url} - uploading file: {file_path}")

        try:
            data = aiohttp.FormData()

            async with aiofiles.open(file_path, 'rb') as f:
                file_content: bytes = await f.read()
                data.add_field('file',
                               file_content,
                               filename=Path(file_path).name,
                               content_type='application/octet-stream')

            headers: Dict[str, str] = self._get_headers()
            async with self._session.post(full_url, headers=headers, data=data, params=params) as response:
                if response.status in (401, 403):
                    raise AuthenticationError(
                        "Authentication failed",
                        status_code=response.status,
                        endpoint=url
                    )

                if response.status != 200:
                    error_text: str = await response.text()
                    if is_external:
                        raise UploadError(f"Upload failed - HTTP {response.status}: {error_text}")
                    raise RequestError(
                        f"HTTP {response.status}: {error_text}",
                        status_code=response.status,
                        endpoint=url,
                        response_text=error_text
                    )

                try:
                    return await response.json()
                except Exception as e:
                    text: str = await response.text()
                    if is_external:
                        raise UploadError(f"Invalid JSON response after upload: {text}")
                    raise RequestError(
                        f"Invalid JSON response: {text}",
                        status_code=response.status,
                        endpoint=url,
                        original_error=e
                    )

        except (AuthenticationError, RequestError, UploadError):
            raise
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            if is_external:
                raise UploadError(f"File upload failed: {str(e)}")
            raise RequestError(
                f"File upload failed: {str(e)}",
                endpoint=url,
                original_error=e
            )

    # -------------------- Image Detection --------------------

    @with_retry(max_attempts=3, delay=1.0)
    async def predict_image_url(self, image_url: str, threshold: float = 0.7) -> ImageDetectionResult:
        """
        Detect pornographic content from an image URL.

        Args:
            image_url: Image URL
            threshold: Decision threshold (default: 0.7)

        Returns:
            ImageDetectionResult: Detection result

        Raises:
            APIResponseError: If the request fails
        """
        params: Dict[str, str] = {"image_url": image_url, "threshold": str(threshold)}
        data: Dict[str, Any] = await self._make_request("GET", "/predict_image_url", params=params)
        return self._parse_image_response(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def predict_image_upload(self, image_path: str, threshold: float = 0.7) -> ImageDetectionResult:
        """
        Detect pornographic content from a local image file.

        Args:
            image_path: Path to local image file
            threshold: Decision threshold (default: 0.7)

        Returns:
            ImageDetectionResult: Detection result

        Raises:
            APIResponseError: If the request fails
            UploadError: If the file upload fails
        """
        params: Dict[str, str] = {"threshold": str(threshold)}
        data: Dict[str, Any] = await self._make_file_request("predict_image_upload", image_path, params)
        return self._parse_image_response(data)

    # -------------------- Video Detection --------------------

    @with_retry(max_attempts=3, delay=1.0)
    async def predict_video_url(self, video_url: str, config: Optional[VideoAnalysisConfig] = None) -> VideoDetectionResult:
        """
        Detect pornographic content from a video URL.

        Args:
            video_url: Video URL
            config: Analysis configuration

        Returns:
            VideoDetectionResult: Detection result

        Raises:
            APIResponseError: If the request fails
        """
        if config is None:
            config = VideoAnalysisConfig()

        params: Dict[str, str] = {"video_url": video_url}
        params.update(config.to_params())

        data: Dict[str, Any] = await self._make_request("GET", "/predict_video_url", params=params)
        return self._parse_video_response(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def request_video_upload_url(self, config: Optional[VideoAnalysisConfig] = None) -> UploadUrl:
        """
        Request an upload URL for video analysis.

        Args:
            config: Analysis configuration

        Returns:
            UploadUrl: Upload URL with key

        Raises:
            APIResponseError: If the request fails
        """
        if config is None:
            config = VideoAnalysisConfig()

        params: Dict[str, str] = config.to_params()
        data: Dict[str, Any] = await self._make_request("GET", "/request_video_upload_url", params=params)
        return self._parse_upload_url_response(data)

    @with_retry(max_attempts=2, delay=2.0)
    async def upload_video_and_analyze(self, video_path: str, config: Optional[VideoAnalysisConfig] = None) -> VideoDetectionResult:
        """
        Upload a video and perform analysis.

        Args:
            video_path: Path to local video file
            config: Analysis configuration

        Returns:
            VideoDetectionResult: Detection result

        Raises:
            APIResponseError: If the request fails
            UploadError: If the file upload fails
        """
        logger.info("Requesting video upload URL...")
        upload_info: UploadUrl = await self.request_video_upload_url(config)

        logger.info(f"Uploading video to: {upload_info.url}")
        data: Dict[str, Any] = await self._make_file_request(upload_info.url, video_path, is_external=True)

        return self._parse_video_response(data)

    # -------------------- Batch Analysis --------------------

    async def _process_batch_analysis(
        self,
        items: List[str],
        analysis_func: Callable,
        item_name: str
    ) -> List[Any]:
        """Generic handler for batch analysis"""
        results: List[Any] = []
        for i, item in enumerate(items):
            try:
                result = await analysis_func(item)
                results.append(result)
                logger.info(f"Processed {item_name} {i + 1}: {item} - Safe: {result.is_safe}")
            except Exception as e:
                logger.error(f"Failed to analyze {item_name} {i + 1} ({item}): {e}")
                if 'image' in item_name.lower():
                    results.append(ImageDetectionResult(success=False))
                else:
                    results.append(VideoDetectionResult(success=False))

        return results

    async def batch_analyze_images(self, image_urls: List[str], threshold: float = 0.7) -> List[ImageDetectionResult]:
        """Analyze multiple images from URLs"""

        async def analyze_single_image(url: str) -> ImageDetectionResult:
            return await self.predict_image_url(url, threshold)

        return await self._process_batch_analysis(image_urls, analyze_single_image, "Image")

    async def batch_analyze_videos(self, video_urls: List[str], config: Optional[VideoAnalysisConfig] = None) -> List[VideoDetectionResult]:
        """Analyze multiple videos from URLs"""

        async def analyze_single_video(url: str) -> VideoDetectionResult:
            return await self.predict_video_url(url, config)

        return await self._process_batch_analysis(video_urls, analyze_single_video, "Video")
