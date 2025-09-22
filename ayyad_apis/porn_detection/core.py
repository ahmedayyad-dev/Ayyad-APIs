import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import aiofiles
import aiohttp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# استثناءات مخصصة
class DetectionError(Exception):
    """خطأ عند فشل عملية الكشف"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Detection failed: {reason}")


class APIResponseError(Exception):
    """خطأ عندما الAPI مايرجعش 200 أو يرجع رسالة خطأ"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(f"API Error: {message}")


class UploadError(Exception):
    """خطأ عند فشل رفع الملف"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Upload failed: {reason}")


# إعدادات تحليل الفيديو
@dataclass
class VideoAnalysisConfig:
    """إعدادات تحليل الفيديو"""
    start_sec: float = 0.0
    duration_sec: float = 0.0  # 0.0 = تحليل كامل
    thresh_high: float = 0.70
    thresh_low: float = 0.60
    min_hit_duration: float = 1.0
    min_ratio: float = 0.02
    smooth_window: int = 5

    def to_params(self) -> Dict[str, str]:
        """تحويل الإعدادات إلى parameters للAPI"""
        return {
            "start_sec": str(self.start_sec),
            "duration_sec": str(self.duration_sec),
            "thresh_high": str(self.thresh_high),
            "thresh_low": str(self.thresh_low),
            "min_hit_duration": str(self.min_hit_duration),
            "min_ratio": str(self.min_ratio),
            "smooth_window": str(self.smooth_window)
        }


# نتيجة كشف الصورة
@dataclass
class ImageDetectionResult:
    """نتيجة كشف المحتوى في الصورة"""
    label: str = "SFW"  # "NSFW" أو "SFW"
    nsfw_prob: float = 0.0
    threshold: float = 0.7
    success: bool = True

    @property
    def is_nsfw(self) -> bool:
        """هل المحتوى غير آمن؟"""
        return self.label == "NSFW"

    @property
    def is_safe(self) -> bool:
        """هل المحتوى آمن؟"""
        return self.label == "SFW"

    @property
    def confidence_percentage(self) -> str:
        """نسبة الثقة كنسبة مئوية"""
        return f"{self.nsfw_prob * 100:.1f}%"

    @property
    def safety_level(self) -> str:
        """مستوى الأمان"""
        if self.is_safe:
            return "آمن"
        elif self.nsfw_prob >= 0.9:
            return "خطر عالي"
        elif self.nsfw_prob >= 0.7:
            return "خطر متوسط"
        else:
            return "خطر منخفض"


# معلومات العتبات المستخدمة في تحليل الفيديو
@dataclass
class VideoThresholds:
    """العتبات المستخدمة في تحليل الفيديو"""
    thresh_high: float = 0.8
    thresh_low: float = 0.7
    min_hit_duration: float = 1.0
    min_ratio: float = 0.02


# إحصائيات تحليل الفيديو
@dataclass
class VideoStats:
    """إحصائيات مفصلة لتحليل الفيديو"""
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
        """أقصى احتمالية كنسبة مئوية"""
        return f"{self.max_prob * 100:.1f}%"

    @property
    def avg_prob_percentage(self) -> str:
        """متوسط الاحتمالية كنسبة مئوية"""
        return f"{self.avg_prob * 100:.1f}%"

    @property
    def ratio_above_percentage(self) -> str:
        """نسبة المحتوى فوق العتبة كنسبة مئوية"""
        return f"{self.ratio_above * 100:.1f}%"

    @property
    def total_duration_formatted(self) -> str:
        """مدة الفيديو الكاملة منسقة"""
        return self._format_duration(self.total_duration)

    @property
    def total_above_duration_formatted(self) -> str:
        """مدة المحتوى غير الآمن منسقة"""
        return self._format_duration(self.total_above_duration)

    @property
    def max_streak_formatted(self) -> str:
        """أطول فترة متواصلة منسقة"""
        return self._format_duration(self.max_streak)

    def _format_duration(self, seconds: float) -> str:
        """تنسيق الوقت إلى HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# نتيجة كشف الفيديو
@dataclass
class VideoDetectionResult:
    """نتيجة كشف المحتوى في الفيديو"""
    nsfw: bool = False
    reason: str = ""
    thresholds: Optional[VideoThresholds] = None
    stats: Optional[VideoStats] = None
    success: bool = True

    @property
    def is_nsfw(self) -> bool:
        """هل المحتوى غير آمن؟"""
        return self.nsfw

    @property
    def is_safe(self) -> bool:
        """هل المحتوى آمن؟"""
        return not self.nsfw

    @property
    def safety_level(self) -> str:
        """مستوى الأمان بناءً على الإحصائيات"""
        if self.is_safe:
            return "آمن"
        elif self.stats and self.stats.max_prob >= 0.9:
            return "خطر عالي"
        elif self.stats and self.stats.max_prob >= 0.7:
            return "خطر متوسط"
        else:
            return "خطر منخفض"


# معلومات رابط الرفع
@dataclass
class UploadUrl:
    """رابط رفع الفيديو"""
    url: str
    key: str = ""

    def __post_init__(self):
        """استخراج المفتاح من الرابط"""
        if "key=" in self.url:
            self.key = self.url.split("key=")[1].split("&")[0]


class PornDetectionAPI:
    """API wrapper لكشف المحتوى الإباحي"""

    def __init__(self, api_key: str, timeout: int = 60, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self._base_url = "https://porn-detection-api.p.rapidapi.com"
        self._headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "porn-detection-api.p.rapidapi.com"
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            timeout=self._timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    def _parse_image_response(self, data: Dict[str, Any]) -> ImageDetectionResult:
        """تحليل استجابة كشف الصورة"""
        try:
            return ImageDetectionResult(
                label=data.get("label", "SFW"),
                nsfw_prob=data.get("nsfw_prob", 0.0),
                threshold=data.get("threshold", 0.7),
                success=True
            )
        except Exception as e:
            logger.warning(f"Failed to parse image response: {e}")
            return ImageDetectionResult(success=False)

    def _parse_video_response(self, data: Dict[str, Any]) -> VideoDetectionResult:
        """تحليل استجابة كشف الفيديو"""
        try:
            # تحليل العتبات
            thresholds_data = data.get("thresholds", {})
            thresholds = VideoThresholds(
                thresh_high=thresholds_data.get("thresh_high", 0.8),
                thresh_low=thresholds_data.get("thresh_low", 0.7),
                min_hit_duration=thresholds_data.get("min_hit_duration", 1.0),
                min_ratio=thresholds_data.get("min_ratio", 0.02)
            )

            # تحليل الإحصائيات
            stats_data = data.get("stats", {})
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
        """تحليل استجابة طلب رابط الرفع"""
        try:
            url = data.get("url", "")
            return UploadUrl(url=url)
        except Exception as e:
            logger.error(f"Failed to parse upload URL response: {e}")
            raise APIResponseError(f"Invalid upload URL response: {e}")

    async def _request_get(self, endpoint: str, params: Dict[str, str]) -> Dict[str, Any]:
        """إرسال طلب GET"""
        url = f"{self._base_url}/{endpoint}"
        logger.info(f"GET {url} with params: {params}")

        try:
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIResponseError(f"HTTP {response.status}: {error_text}")

                try:
                    return await response.json()
                except Exception as e:
                    text = await response.text()
                    raise APIResponseError(f"Invalid JSON response: {text}")

        except APIResponseError:
            raise
        except Exception as e:
            raise APIResponseError(f"Request failed: {str(e)}")

    async def _request_post_multipart(self, url: str, file_path: str, params: Dict[str, str] = None) -> Dict[str, Any]:
        """إرسال طلب POST مع رفع ملف"""
        logger.info(f"POST {url} - uploading file: {file_path}")

        if not Path(file_path).exists():
            raise UploadError(f"File not found: {file_path}")

        try:
            data = aiohttp.FormData()

            # إضافة الملف
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
                data.add_field('file',
                               file_content,
                               filename=Path(file_path).name,
                               content_type='application/octet-stream')

            # إرسال الطلب
            async with self._session.post(url, data=data, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise UploadError(f"Upload failed - HTTP {response.status}: {error_text}")

                try:
                    return await response.json()
                except Exception as e:
                    text = await response.text()
                    raise UploadError(f"Invalid JSON response after upload: {text}")

        except (UploadError, APIResponseError):
            raise
        except Exception as e:
            raise UploadError(f"Upload request failed: {str(e)}")

    async def _request_post_internal(self, endpoint: str, file_path: str, params: Dict[str, str]) -> Dict[str, Any]:
        """إرسال طلب POST للendpoints الداخلية"""
        url = f"{self._base_url}/{endpoint}"
        logger.info(f"POST {url} - uploading file: {file_path}")

        if not Path(file_path).exists():
            raise UploadError(f"File not found: {file_path}")

        try:
            data = aiohttp.FormData()

            # إضافة الملف
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
                data.add_field('file',
                               file_content,
                               filename=Path(file_path).name,
                               content_type='application/octet-stream')

            # إرسال الطلب
            async with self._session.post(url, data=data, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIResponseError(f"HTTP {response.status}: {error_text}")

                try:
                    return await response.json()
                except Exception as e:
                    text = await response.text()
                    raise APIResponseError(f"Invalid JSON response: {text}")

        except APIResponseError:
            raise
        except Exception as e:
            raise APIResponseError(f"Request failed: {str(e)}")

    # ==================== Image Detection Methods ====================

    async def predict_image_url(self, image_url: str, threshold: float = 0.7) -> ImageDetectionResult:
        """
        كشف المحتوى الإباحي من رابط صورة.

        Args:
            image_url (str): رابط الصورة
            threshold (float): العتبة للقرار (افتراضي: 0.7)

        Returns:
            ImageDetectionResult: نتيجة كشف المحتوى

        Raises:
            APIResponseError: إذا فشل الطلب
        """
        params = {
            "image_url": image_url,
            "threshold": str(threshold)
        }

        data = await self._request_get("predict_image_url", params)
        return self._parse_image_response(data)

    async def predict_image_upload(self, image_path: str, threshold: float = 0.7) -> ImageDetectionResult:
        """
        كشف المحتوى الإباحي من ملف صورة محلي.

        Args:
            image_path (str): مسار الصورة المحلية
            threshold (float): العتبة للقرار (افتراضي: 0.7)

        Returns:
            ImageDetectionResult: نتيجة كشف المحتوى

        Raises:
            APIResponseError: إذا فشل الطلب
            UploadError: إذا فشل رفع الملف
        """
        params = {"threshold": str(threshold)}
        data = await self._request_post_internal("predict_image_upload", image_path, params)
        return self._parse_image_response(data)

    # ==================== Video Detection Methods ====================

    async def predict_video_url(self, video_url: str, config: VideoAnalysisConfig = None) -> VideoDetectionResult:
        """
        كشف المحتوى الإباحي من رابط فيديو.

        Args:
            video_url (str): رابط الفيديو
            config (VideoAnalysisConfig): إعدادات التحليل

        Returns:
            VideoDetectionResult: نتيجة كشف المحتوى

        Raises:
            APIResponseError: إذا فشل الطلب
        """
        if config is None:
            config = VideoAnalysisConfig()

        params = {"video_url": video_url}
        params.update(config.to_params())

        data = await self._request_get("predict_video_url", params)
        return self._parse_video_response(data)

    async def request_video_upload_url(self, config: VideoAnalysisConfig = None) -> UploadUrl:
        """
        طلب رابط لرفع فيديو للتحليل.

        Args:
            config (VideoAnalysisConfig): إعدادات التحليل

        Returns:
            UploadUrl: رابط الرفع مع المفتاح

        Raises:
            APIResponseError: إذا فشل الطلب
        """
        if config is None:
            config = VideoAnalysisConfig()

        params = config.to_params()
        data = await self._request_get("request_video_upload_url", params)
        return self._parse_upload_url_response(data)

    async def upload_video_and_analyze(self, video_path: str,
                                       config: VideoAnalysisConfig = None) -> VideoDetectionResult:
        """
        رفع فيديو وتحليله (العملية الكاملة).

        Args:
            video_path (str): مسار الفيديو المحلي
            config (VideoAnalysisConfig): إعدادات التحليل

        Returns:
            VideoDetectionResult: نتيجة كشف المحتوى

        Raises:
            APIResponseError: إذا فشل الطلب
            UploadError: إذا فشل رفع الملف
        """
        # الحصول على رابط الرفع
        logger.info("طلب رابط رفع الفيديو...")
        upload_info = await self.request_video_upload_url(config)

        # رفع الفيديو
        logger.info(f"رفع الفيديو إلى: {upload_info.url}")
        data = await self._request_post_multipart(upload_info.url, video_path)

        return self._parse_video_response(data)

    # ==================== Batch Operations ====================

    async def _process_batch_analysis(self, items: list, analysis_func, item_name: str) -> list:
        """معالجة موحدة للتحليل المتعدد"""
        results = []
        for i, item in enumerate(items):
            try:
                result = await analysis_func(item)
                results.append(result)
                logger.info(f"تم تحليل {item_name} {i + 1}: {item} - آمن: {result.is_safe}")
            except Exception as e:
                logger.error(f"فشل تحليل {item_name} {i + 1} ({item}): {e}")
                # إنشاء نتيجة فاشلة حسب نوع التحليل
                if 'image' in item_name.lower():
                    results.append(ImageDetectionResult(success=False))
                else:
                    results.append(VideoDetectionResult(success=False))

        return results

    async def batch_analyze_images(self, image_urls: list, threshold: float = 0.7) -> list[ImageDetectionResult]:
        """
        تحليل عدة صور من روابط.

        Args:
            image_urls (list): قائمة روابط الصور
            threshold (float): العتبة للقرار

        Returns:
            list[ImageDetectionResult]: قائمة نتائج التحليل
        """

        async def analyze_single_image(url):
            return await self.predict_image_url(url, threshold)

        return await self._process_batch_analysis(image_urls, analyze_single_image, "صورة")

    async def batch_analyze_videos(self, video_urls: list, config: VideoAnalysisConfig = None) -> list[VideoDetectionResult]:
        """
        تحليل عدة فيديوهات من روابط.

        Args:
            video_urls (list): قائمة روابط الفيديوهات
            config (VideoAnalysisConfig): إعدادات التحليل

        Returns:
            list[VideoDetectionResult]: قائمة نتائج التحليل
        """

        async def analyze_single_video(url):
            return await self.predict_video_url(url, config)

        return await self._process_batch_analysis(video_urls, analyze_single_video, "فيديو")

