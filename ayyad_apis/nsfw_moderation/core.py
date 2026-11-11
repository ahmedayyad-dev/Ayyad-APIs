"""
NSFW Image & Video Moderation API wrapper.

This module provides an async interface to the NSFW Image & Video Moderation API
hosted on RapidAPI (nsfw-image-video-moderation-api.p.rapidapi.com). It allows
users to classify images and videos for adult/NSFW content via direct URLs or
file uploads, handling HTTP 451 detection responses transparently as valid results.

Author: Ahmed Ayyad
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utils import BaseResponse, InvalidInputError
from ..porn_detection.core import (
    PornDetectionAPI,
    ImageDetectionResult,
    VideoDetectionResult,
    VideoAnalysisConfig,
    UploadUrl,
    VideoStats,
    VideoThresholds,
    VideoAnalysis,
    VideoSegment,
    DetectionError as NSFWModerationError,
    APIResponseError,
    UploadError,
)

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
_DEFAULT_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".m4v", ".mkv", ".webm"})


@dataclass
class DirectoryScanResult(BaseResponse):
    """Result of scanning a local filesystem folder for NSFW media."""

    directory: str = ""
    total_files: int = 0
    safe_files: List[str] = field(default_factory=list)
    flagged_files: List[Tuple[str, str]] = field(default_factory=list)  # (file_path, reason)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)   # (file_path, error_message)

    @property
    def is_clean(self) -> bool:
        """True if no files were flagged as NSFW and no failures occurred."""
        return len(self.flagged_files) == 0 and len(self.failed_files) == 0

    @property
    def flagged_count(self) -> int:
        return len(self.flagged_files)

    @property
    def safe_count(self) -> int:
        return len(self.safe_files)


@dataclass
class QuarantineResult(BaseResponse):
    """Result of scanning and isolating flagged NSFW media."""

    scan_result: DirectoryScanResult
    quarantined_files: List[str] = field(default_factory=list)

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined_files)


class NSFWModerationAPI(PornDetectionAPI):
    """
    Async client for NSFW Image & Video Moderation API on RapidAPI.

    API Host: ``nsfw-image-video-moderation-api.p.rapidapi.com``

    Supports:
    - Image classification from direct HTTP/HTTPS URL (``predict_image_url``)
    - Image classification via file upload (``predict_image_upload``)
    - Video classification from direct HTTP/HTTPS URL (``predict_video_url``)
    - Temporary 24h upload URL generation (``request_video_upload_url``)
    - Local video upload & classification (``upload_video_and_predict``)
    - One-liner boolean safety check (``is_safe``)
    - Directory media audit scanner (``scan_directory``)
    - Automated file quarantine (``quarantine_directory``)
    - Batch moderation (``moderate_batch``)

    HTTP 451 responses returned when NSFW content is detected are parsed as
    valid result objects with ``is_nsfw=True`` rather than raising exceptions.

    Example:
        async with NSFWModerationAPI(api_key="your_key") as api:
            # One-liner safety check
            if not await api.is_safe("photo.jpg"):
                print("Content rejected: NSFW detected")

            # Scan entire folder of user uploads
            audit = await api.scan_directory("./uploads")
            print(f"Flagged {audit.flagged_count} files out of {audit.total_files}")
    """

    BASE_URL = "https://nsfw-image-video-moderation-api.p.rapidapi.com"
    DEFAULT_HOST = "nsfw-image-video-moderation-api.p.rapidapi.com"

    async def upload_video_and_predict(
        self,
        video_path: Union[str, Path],
        config: Optional[VideoAnalysisConfig] = None
    ) -> VideoDetectionResult:
        """
        Request an upload URL, upload the video file, and return the moderation result.

        Args:
            video_path: Local path to the video file.
            config: Optional VideoAnalysisConfig for threshold and duration settings.

        Returns:
            VideoDetectionResult with moderation verdict and stats.
        """
        return await self.upload_video_and_analyze(video_path, config=config)

    # -------------------- Content Safety Workflows --------------------

    async def is_safe(
        self,
        target: Union[str, Path],
        threshold: float = 0.70,
    ) -> bool:
        """
        Quick boolean gatekeeper: returns True if content is safe, False if flagged as NSFW.

        Accepts any image/video URL or local file path transparently.

        Args:
            target: Image or video URL (http/https), or local file path.
            threshold: Confidence threshold above which media is considered NSFW (default: 0.70).

        Returns:
            True if content is SFW/safe, False if flagged as NSFW.
        """
        target_str = str(target).strip()
        is_url = target_str.startswith("http://") or target_str.startswith("https://")

        # Determine extension if present
        ext = Path(target_str).suffix.lower()
        is_video = ext in _DEFAULT_VIDEO_EXTENSIONS

        try:
            if is_url:
                if is_video:
                    res = await self.predict_video_url(target_str)
                    return not res.nsfw
                else:
                    res = await self.predict_image_url(target_str, threshold=threshold)
                    return not res.is_nsfw
            else:
                local_path = Path(target)
                if not local_path.exists():
                    raise InvalidInputError(f"Local file does not exist: {local_path}")

                if is_video:
                    res = await self.upload_video_and_predict(local_path)
                    return not res.nsfw
                else:
                    res = await self.predict_image_upload(local_path, threshold=threshold)
                    return not res.is_nsfw
        except Exception as e:
            logger.error(f"Error checking safety for {target}: {e}")
            raise

    async def scan_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        threshold: float = 0.70,
        concurrency: int = 3,
        image_extensions: Optional[Tuple[str, ...]] = None,
        video_extensions: Optional[Tuple[str, ...]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> DirectoryScanResult:
        """
        Concurrently audit all media files in a local folder for NSFW content.

        Args:
            directory_path: Path to the directory to scan.
            recursive: If True, scans subdirectories recursively.
            threshold: Classification threshold (default: 0.70).
            concurrency: Maximum concurrent upload/classification workers (default: 3).
            image_extensions: Tuple of file extensions considered images.
            video_extensions: Tuple of file extensions considered videos.
            progress_callback: Optional callback receiving (file_path, processed_count, total).

        Returns:
            DirectoryScanResult detailing safe, flagged, and failed files.
        """
        dir_p = Path(directory_path)
        if not dir_p.is_dir():
            raise InvalidInputError(f"Directory not found: {dir_p}")

        img_exts = frozenset(image_extensions) if image_extensions else _DEFAULT_IMAGE_EXTENSIONS
        vid_exts = frozenset(video_extensions) if video_extensions else _DEFAULT_VIDEO_EXTENSIONS
        all_exts = img_exts | vid_exts

        if recursive:
            files = [p for p in dir_p.rglob("*") if p.is_file() and p.suffix.lower() in all_exts]
        else:
            files = [p for p in dir_p.glob("*") if p.is_file() and p.suffix.lower() in all_exts]

        result = DirectoryScanResult(directory=str(dir_p), total_files=len(files))
        if not files:
            return result

        sem = asyncio.Semaphore(max(1, concurrency))
        lock = asyncio.Lock()
        completed = 0
        total = len(files)

        async def _check(file_path: Path) -> None:
            nonlocal completed
            async with sem:
                try:
                    ext = file_path.suffix.lower()
                    if ext in vid_exts:
                        verdict = await self.upload_video_and_predict(file_path)
                        is_flagged = verdict.nsfw
                        reason = verdict.reason or "NSFW video detected"
                    else:
                        verdict = await self.predict_image_upload(file_path, threshold=threshold)
                        is_flagged = verdict.is_nsfw
                        reason = f"Label: {verdict.label}, Prob: {verdict.nsfw_prob:.2f}"

                    async with lock:
                        if is_flagged:
                            result.flagged_files.append((str(file_path), reason))
                        else:
                            result.safe_files.append(str(file_path))
                except Exception as e:
                    logger.error(f"Error scanning {file_path}: {e}")
                    async with lock:
                        result.failed_files.append((str(file_path), str(e)))
                finally:
                    async with lock:
                        completed += 1
                        if progress_callback:
                            progress_callback(str(file_path), completed, total)

        await asyncio.gather(*[_check(f) for f in files])
        return result

    async def quarantine_directory(
        self,
        directory_path: Union[str, Path],
        quarantine_dir: Union[str, Path],
        recursive: bool = True,
        threshold: float = 0.70,
        concurrency: int = 3,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> QuarantineResult:
        """
        Scan a directory and automatically isolate any flagged NSFW files into a quarantine folder.

        Args:
            directory_path: Source directory containing media files.
            quarantine_dir: Destination folder where flagged files will be moved.
            recursive: Whether to scan subdirectories.
            threshold: NSFW confidence threshold.
            concurrency: Maximum concurrent scan tasks.
            progress_callback: Optional progress callback.

        Returns:
            QuarantineResult detailing the scan and the list of isolated file paths.
        """
        q_dir = Path(quarantine_dir)
        q_dir.mkdir(parents=True, exist_ok=True)

        scan = await self.scan_directory(
            directory_path=directory_path,
            recursive=recursive,
            threshold=threshold,
            concurrency=concurrency,
            progress_callback=progress_callback,
        )

        quarantined: List[str] = []
        for file_str, _ in scan.flagged_files:
            source = Path(file_str)
            if source.exists():
                dest = q_dir / source.name
                # Avoid collision if same name exists
                if dest.exists():
                    dest = q_dir / f"{source.stem}_{source.stat().st_mtime_ns}{source.suffix}"
                shutil.move(str(source), str(dest))
                quarantined.append(str(dest))

        return QuarantineResult(scan_result=scan, quarantined_files=quarantined)

    async def moderate_batch(
        self,
        targets: List[Union[str, Path]],
        threshold: float = 0.70,
        concurrency: int = 3,
    ) -> List[Tuple[Union[str, Path], bool, Any]]:
        """
        Concurrently moderate a list of image/video URLs or local files.

        Args:
            targets: List of URLs or file paths.
            threshold: Confidence threshold.
            concurrency: Maximum concurrent workers.

        Returns:
            List of tuples: ``(target, is_safe, result_or_exception)``.
        """
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _worker(t: Union[str, Path]) -> Tuple[Union[str, Path], bool, Any]:
            async with sem:
                try:
                    safe = await self.is_safe(t, threshold=threshold)
                    return (t, safe, None)
                except Exception as e:
                    return (t, False, e)

        return await asyncio.gather(*[_worker(t) for t in targets])
