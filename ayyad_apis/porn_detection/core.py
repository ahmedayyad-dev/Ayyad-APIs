"""
Porn Detection API wrapper for detecting NSFW content in images and videos.

This module provides a simple async interface to interact with Porn Detection API
through RapidAPI, allowing users to detect pornographic content in images and videos.

Author: Ahmed Ayyad
"""

import json
import logging
from dataclasses import dataclass, asdict, field
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


# ==================== HTTP 451 ====================

def _parse_451_or_raise(error: ClientError) -> Dict[str, Any]:
    """Return an HTTP 451 response as a valid NSFW result.

    The Porn Detection API returns HTTP 451 (Unavailable For Legal Reasons)
    when NSFW content is detected. Every endpoint returns the full JSON body
    with 451, so the body is used as-is when present and only synthesised when
    it is missing. Either way, HTTP 451 means the content was flagged, so treat
    it as a result instead of an error.
    """
    if error.status_code == 451:
        if error.response_text:
            try:
                data = json.loads(error.response_text)
            except ValueError:
                data = None
            if isinstance(data, dict) and data:
                data["nsfw"] = True
                data["label"] = "NSFW"
                data["nsfw_prob"] = 1.0
                data["is_nsfw"] = True
                data["policy"] = "BLOCK"
                return data
        return {
            "nsfw": True,
            "is_nsfw": True,
            "label": "NSFW",
            "nsfw_prob": 1.0,
            "policy": "BLOCK",
        }
    raise error


# ==================== Video Analysis Configuration ====================

@dataclass
class VideoAnalysisConfig(BaseResponse):
    """Configuration for video analysis.

    Defaults mirror the server's own defaults.
    """

    start_sec: float = 0.0
    duration_sec: Optional[float] = None  # None = analyze the entire video
    thresh_high: float = 0.80
    thresh_low: float = 0.70
    min_hit_duration: float = 1.0
    min_ratio: float = 0.02

    def to_params(self) -> Dict[str, str]:
        """Convert settings into API parameters.

        ``duration_sec`` is only sent when explicitly set, because the server
        distinguishes "not provided" (``None``) from a numeric value.
        """
        params = {
            "start_sec": str(self.start_sec),
            "thresh_high": str(self.thresh_high),
            "thresh_low": str(self.thresh_low),
            "min_hit_duration": str(self.min_hit_duration),
            "min_ratio": str(self.min_ratio),
        }
        if self.duration_sec is not None:
            params["duration_sec"] = str(self.duration_sec)
        return params

    # to_dict() and to_json() inherited from BaseResponse


# ==================== Image Detection Result ====================

@dataclass
class ImageDetectionResult(BaseResponse):
    """Result of image content detection.

    Fields mirror the API's ``classify_image_prob`` payload exactly.
    """

    label: str = "SFW"  # "NSFW" or "SFW"
    nsfw_prob: float = 0.0
    threshold: float = 0.7
    is_nsfw: bool = False
    sfw_prob: float = 1.0
    confidence: float = 0.0
    confidence_level: Optional[str] = None  # very_low|low|medium|high|very_high
    distance_from_threshold: float = 0.0
    policy: Optional[str] = None  # ALLOW|REVIEW|BLOCK
    scores: Optional[Dict[str, float]] = None
    bbox: Optional[Dict[str, Any]] = None
    crops: Optional[Any] = None
    crop_decision: Optional[Any] = None
    crop_scores: Optional[Dict[str, float]] = None
    elapsed: Optional[float] = None
    success: bool = True

    @property
    def is_safe(self) -> bool:
        """Check if content is safe."""
        return not self.is_nsfw

    @property
    def confidence_percentage(self) -> str:
        """Confidence level as percentage."""
        return f"{self.nsfw_prob * 100:.1f}%"

    @property
    def safety_level(self) -> str:
        """Human-readable safety summary derived from the probability."""
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
        data = super().to_dict()
        data.update({
            "is_safe": self.is_safe,
            "confidence_percentage": self.confidence_percentage,
            "safety_level": self.safety_level,
        })
        return data

    # to_json() inherited from BaseResponse


# ==================== Video Result Models ====================

@dataclass
class VideoThresholds(BaseResponse):
    """Thresholds used in video analysis (mirrors the API's ``thresholds``)."""
    thresh_high: float = 0.8
    thresh_low: float = 0.7
    min_hit_duration: float = 1.0
    min_ratio: float = 0.02
    min_hit_duration_effective: Optional[float] = None

    # to_dict() and to_json() inherited from BaseResponse


@dataclass
class VideoSegment(BaseResponse):
    """A contiguous run of flagged frames (mirrors ``video.segments[]``)."""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0


@dataclass
class VideoAnalysis(BaseResponse):
    """Video-level analysis metadata (mirrors the API's ``video`` object).

    Note:
        The API reports total/analysed duration and the frame sampling grid
        here, not inside ``stats``.
    """
    duration: float = 0.0
    sampled_duration: float = 0.0
    frames_checked: int = 0
    sample_step: float = 0.0
    segments: List[VideoSegment] = field(default_factory=list)

    @property
    def duration_formatted(self) -> str:
        """Format total duration as HH:MM:SS."""
        return self._format_duration(self.duration)

    def _format_duration(self, seconds: float) -> str:
        hours: int = int(seconds // 3600)
        minutes: int = int((seconds % 3600) // 60)
        secs: int = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["duration_formatted"] = self.duration_formatted
        return data


@dataclass
class VideoStats(BaseResponse):
    """Probability statistics for video analysis (mirrors the API's ``stats``).

    Note:
        Total duration and the sampling grid live in :class:`VideoAnalysis`,
        not here.
    """
    max_prob: float = 0.0
    avg_prob: float = 0.0
    total_above_duration: float = 0.0
    ratio_above: float = 0.0
    max_streak: float = 0.0

    @property
    def max_prob_percentage(self) -> str:
        """Maximum probability as percentage."""
        return f"{self.max_prob * 100:.1f}%"

    @property
    def avg_prob_percentage(self) -> str:
        """Average probability as percentage."""
        return f"{self.avg_prob * 100:.1f}%"

    @property
    def ratio_above_percentage(self) -> str:
        """Ratio above threshold as percentage."""
        return f"{self.ratio_above * 100:.1f}%"

    @property
    def total_above_duration_formatted(self) -> str:
        """Formatted unsafe content duration."""
        return self._format_duration(self.total_above_duration)

    @property
    def max_streak_formatted(self) -> str:
        """Formatted longest continuous unsafe streak."""
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
            "total_above_duration_formatted": self.total_above_duration_formatted,
            "max_streak_formatted": self.max_streak_formatted,
        })
        return data

    # to_json() inherited from BaseResponse


@dataclass
class VideoDetectionResult(BaseResponse):
    """Result of video content detection.

    ``predict_*`` endpoints answer HTTP 451 with this same payload when the
    video is flagged; ``analyze_*`` always answer 200.
    """
    nsfw: bool = False
    reason: str = ""
    policy: Optional[str] = None  # ALLOW|REVIEW|BLOCK
    thresholds: Optional[VideoThresholds] = None
    video: Optional[VideoAnalysis] = None
    stats: Optional[VideoStats] = None
    bbox: Optional[Dict[str, Any]] = None
    worst_frame: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, float]] = None
    crops: Optional[Any] = None
    elapsed: Optional[float] = None
    success: bool = True

    @property
    def is_nsfw(self) -> bool:
        """Check if content is unsafe."""
        return self.nsfw

    @property
    def is_safe(self) -> bool:
        """Check if content is safe."""
        return not self.nsfw

    @property
    def safety_level(self) -> str:
        """Determine safety level based on statistics."""
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
        data = super().to_dict()
        data.update({
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "video": self.video.to_dict() if self.video else None,
            "stats": self.stats.to_dict() if self.stats else None,
            "is_nsfw": self.is_nsfw,
            "safety_level": self.safety_level,
        })
        return data

    # to_json() inherited from BaseResponse


# ==================== Upload URL Information ====================

@dataclass
class UploadUrl(BaseResponse):
    """Video upload URL returned by /request_video_upload_url.

    Note:
        The API builds this URL from its own ``request.base_url``, so it may
        point at an internal host and may require an infrastructure-level
        proxy-secret header that this library does not send. It is only
        reachable when the API is fronted by a gateway that injects it.
    """
    url: str
    key: str = ""
    expires_in: Optional[int] = None

    def __post_init__(self) -> None:
        """Extract key and TTL from the URL."""
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

    Two endpoint families are exposed:

    - ``predict_*`` returns 451 when content is flagged. That is treated as a
      result, not an error, so you always get a populated result object.
    - ``analyze_*`` always returns 200 with the same payload plus per-class
      scores and a bounding box, and never raises 451.

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
                 config: Optional[APIConfig] = None, own_session: bool = False) -> None:
        super().__init__(api_key=api_key, timeout=timeout, config=config, own_session=own_session)
        self._max_retries: int = max_retries
        self._retry_delay: float = retry_delay

    async def _make_request(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        """Make HTTP request, converting an HTTP 451 detection result into a valid response."""
        try:
            return await super()._make_request(method, endpoint, **kwargs)
        except ClientError as e:
            return _parse_451_or_raise(e)

    async def _post_form_data(self, endpoint: str, form_data: Any, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST multipart form data, converting an HTTP 451 detection result into a valid response."""
        try:
            return await super()._post_form_data(endpoint, form_data, params=params)
        except ClientError as e:
            return _parse_451_or_raise(e)

    # __aenter__ and __aexit__ inherited from BaseRapidAPI

    # -------------------- Response Parsers --------------------

    def _parse_image_response(self, data: Dict[str, Any]) -> ImageDetectionResult:
        """Parse image detection response"""
        try:
            return ImageDetectionResult(
                label=data.get("label", "SFW"),
                nsfw_prob=data.get("nsfw_prob", 0.0),
                threshold=data.get("threshold", 0.7),
                is_nsfw=data.get("is_nsfw", False),
                sfw_prob=data.get("sfw_prob", 1.0 - data.get("nsfw_prob", 0.0)),
                confidence=data.get("confidence", 0.0),
                confidence_level=data.get("confidence_level"),
                distance_from_threshold=data.get("distance_from_threshold", 0.0),
                policy=data.get("policy"),
                scores=data.get("scores"),
                bbox=data.get("bbox"),
                crops=data.get("crops"),
                crop_decision=data.get("crop_decision"),
                crop_scores=data.get("crop_scores"),
                elapsed=data.get("elapsed"),
                success=True,
            )
        except Exception as e:
            logger.warning(f"Failed to parse image response: {e}")
            return ImageDetectionResult(success=False)

    @staticmethod
    def _parse_thresholds(data: Dict[str, Any]) -> VideoThresholds:
        return VideoThresholds(
            thresh_high=data.get("thresh_high", 0.8),
            thresh_low=data.get("thresh_low", 0.7),
            min_hit_duration=data.get("min_hit_duration", 1.0),
            min_ratio=data.get("min_ratio", 0.02),
            min_hit_duration_effective=data.get("min_hit_duration_effective"),
        )

    @staticmethod
    def _parse_video_block(data: Dict[str, Any]) -> VideoAnalysis:
        return VideoAnalysis(
            duration=data.get("duration", 0.0),
            sampled_duration=data.get("sampled_duration", 0.0),
            frames_checked=data.get("frames_checked", 0),
            sample_step=data.get("sample_step", 0.0),
            segments=[VideoSegment(**s) for s in data.get("segments", []) if isinstance(s, dict)],
        )

    @staticmethod
    def _parse_stats(data: Dict[str, Any]) -> VideoStats:
        return VideoStats(
            max_prob=data.get("max_prob", 0.0),
            avg_prob=data.get("avg_prob", 0.0),
            total_above_duration=data.get("total_above_duration", 0.0),
            ratio_above=data.get("ratio_above", 0.0),
            max_streak=data.get("max_streak", 0.0),
        )

    def _parse_video_response(self, data: Dict[str, Any]) -> VideoDetectionResult:
        """Parse video detection response"""
        try:
            return VideoDetectionResult(
                nsfw=data.get("nsfw", False),
                reason=data.get("reason", ""),
                policy=data.get("policy"),
                thresholds=self._parse_thresholds(data.get("thresholds", {})),
                video=self._parse_video_block(data.get("video", {})),
                stats=self._parse_stats(data.get("stats", {})),
                bbox=data.get("bbox"),
                worst_frame=data.get("worst_frame"),
                scores=data.get("scores"),
                crops=data.get("crops"),
                elapsed=data.get("elapsed"),
                success=True,
            )
        except Exception as e:
            logger.warning(f"Failed to parse video response: {e}")
            return VideoDetectionResult(success=False)

    def _parse_upload_url_response(self, data: Dict[str, Any]) -> UploadUrl:
        """Parse response for upload URL request"""
        try:
            url: str = data.get("url", "")
            if not url:
                raise APIResponseError("Upload URL response contained no 'url'")
            return UploadUrl(url=url, expires_in=data.get("expires_in"))
        except APIResponseError:
            raise
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
            url: Full URL (if is_external=True) or endpoint name without leading slash
            file_path: Path to the file to upload
            params: Optional query parameters
            is_external: If True, url is a full upload URL; otherwise an API endpoint

        Returns:
            JSON response as dictionary

        Raises:
            InvalidInputError: If file not found
            AuthenticationError: If authentication fails
            UploadError/RequestError: If upload fails
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        if not Path(file_path).exists():
            raise InvalidInputError(f"File not found: {file_path}")

        logger.info(f"POST {'<external>' if is_external else self.BASE_URL + '/' + url} - uploading file: {file_path}")

        data = aiohttp.FormData()
        async with aiofiles.open(file_path, 'rb') as f:
            file_content: bytes = await f.read()
            data.add_field('file',
                           file_content,
                           filename=Path(file_path).name,
                           content_type='application/octet-stream')

        if is_external:
            # Pre-signed/upload target — keep full manual control over headers
            # and errors, but treat HTTP 451 as a detection result like every
            # other endpoint.
            try:
                # Content-Type MUST be omitted so aiohttp generates the
                # multipart boundary. Sending application/json here makes the
                # server unable to bind the file to its `file` parameter.
                headers: Dict[str, str] = {
                    "x-rapidapi-host": self.rapidapi_host,
                    "x-rapidapi-key": self.api_key,
                }
                if self.config and self.config.extra_headers:
                    headers.update(self.config.extra_headers)

                async with self._session.post(url, headers=headers, data=data, params=params) as response:
                    if response.status in (401, 403):
                        raise AuthenticationError(
                            "Authentication failed",
                            status_code=response.status,
                            endpoint=url
                        )
                    if response.status == 451:
                        error_text: str = await response.text()
                        return _parse_451_or_raise(ClientError(
                            "NSFW content detected",
                            status_code=451,
                            response_text=error_text
                        ))
                    if response.status != 200:
                        error_text = await response.text()
                        raise UploadError(
                            f"Upload failed - HTTP {response.status}: {error_text}",
                            status_code=response.status,
                            response_text=error_text,
                        )
                    try:
                        return await response.json()
                    except (aiohttp.ContentTypeError, ValueError) as e:
                        text: str = await response.text()
                        raise UploadError(f"Invalid JSON response after upload: {text}") from e
            except (AuthenticationError, UploadError):
                raise
            except Exception as e:
                logger.error(f"Upload error: {str(e)}")
                raise UploadError(f"File upload failed: {str(e)}")
        else:
            # Internal API endpoint — delegate to shared _post_form_data
            return await self._post_form_data(f"/{url}", data, params=params)

    # -------------------- Image Detection --------------------

    @with_retry(max_attempts=3, delay=1.0)
    async def predict_image_url(self, image_url: str, threshold: float = 0.7,
                                multi_crop: bool = True) -> ImageDetectionResult:
        """
        Detect pornographic content from an image URL.

        Args:
            image_url: Image URL
            threshold: Decision threshold, 0.01-0.99 (default: 0.7)
            multi_crop: Also score centre crops to reduce false negatives (default: True)

        Returns:
            ImageDetectionResult: Detection result
        """
        params: Dict[str, str] = {"image_url": image_url, "threshold": str(threshold)}
        if not multi_crop:
            params["multi_crop"] = "false"
        data: Dict[str, Any] = await self._make_request("GET", "/predict_image_url", params=params)
        return self._parse_image_response(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def predict_image_upload(self, image_path: str, threshold: float = 0.7,
                                   multi_crop: bool = True) -> ImageDetectionResult:
        """
        Detect pornographic content from a local image file.

        Args:
            image_path: Path to local image file
            threshold: Decision threshold, 0.01-0.99 (default: 0.7).
                Note the server's own default for this endpoint is 0.80; it is
                always sent explicitly so behaviour stays predictable.
            multi_crop: Also score centre crops to reduce false negatives (default: True)

        Returns:
            ImageDetectionResult: Detection result

        Raises:
            UploadError: If the file upload fails
        """
        params: Dict[str, str] = {"threshold": str(threshold)}
        if not multi_crop:
            params["multi_crop"] = "false"
        data: Dict[str, Any] = await self._make_file_request("predict_image_upload", image_path, params)
        return self._parse_image_response(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def analyze_image(self, image_url: str, threshold: float = 0.7,
                            multi_crop: bool = True) -> ImageDetectionResult:
        """
        Full image analysis: per-class scores, policy and the tightest crop box.

        Unlike :meth:`predict_image_url` this always returns HTTP 200, never 451.

        Args:
            image_url: Image URL
            threshold: Decision threshold, 0.01-0.99 (default: 0.7)
            multi_crop: Also score centre crops to reduce false negatives (default: True)
        """
        params: Dict[str, str] = {"image_url": image_url, "threshold": str(threshold)}
        if not multi_crop:
            params["multi_crop"] = "false"
        data: Dict[str, Any] = await self._make_request("GET", "/analyze_image", params=params)
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
        """
        if config is None:
            config = VideoAnalysisConfig()

        params: Dict[str, str] = {"video_url": video_url}
        params.update(config.to_params())

        data: Dict[str, Any] = await self._make_request("GET", "/predict_video_url", params=params)
        return self._parse_video_response(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def analyze_video(self, video_url: str, config: Optional[VideoAnalysisConfig] = None) -> VideoDetectionResult:
        """
        Full video analysis returning segments, policy and a bounding box on the worst frame.

        Unlike :meth:`predict_video_url` this always returns HTTP 200, never 451.

        Args:
            video_url: Video URL
            config: Analysis configuration
        """
        if config is None:
            config = VideoAnalysisConfig()

        params: Dict[str, str] = {"video_url": video_url}
        params.update(config.to_params())

        data: Dict[str, Any] = await self._make_request("GET", "/analyze_video", params=params)
        return self._parse_video_response(data)

    @with_retry(max_attempts=3, delay=1.0)
    async def request_video_upload_url(self, config: Optional[VideoAnalysisConfig] = None) -> UploadUrl:
        """
        Request an upload URL for video analysis.

        Args:
            config: Analysis configuration (stored server-side with the key)

        Returns:
            UploadUrl: Upload URL with key and expiry
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

        Note:
            A flagged (NSFW) video returns a populated result with ``nsfw=True``
            rather than raising.
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