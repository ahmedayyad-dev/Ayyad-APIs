import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from ayyad_apis.nsfw_moderation import (
    NSFWModerationAPI,
    NSFWImageVideoModerationAPI,
    ImageDetectionResult,
    VideoDetectionResult,
    VideoAnalysisConfig,
    UploadUrl,
    DirectoryScanResult,
    QuarantineResult,
)
from ayyad_apis.utils import ClientError, InvalidInputError


@pytest.fixture
def api_key():
    return "test_rapidapi_key"


@pytest.mark.asyncio
async def test_api_aliases_and_constants(api_key):
    assert NSFWImageVideoModerationAPI is NSFWModerationAPI
    client = NSFWModerationAPI(api_key=api_key)
    assert client.BASE_URL == "https://nsfw-image-video-moderation-api.p.rapidapi.com"
    assert client.DEFAULT_HOST == "nsfw-image-video-moderation-api.p.rapidapi.com"
    assert client.rapidapi_host == "nsfw-image-video-moderation-api.p.rapidapi.com"


@pytest.mark.asyncio
async def test_predict_image_url_safe(api_key):
    async with NSFWModerationAPI(api_key=api_key) as client:
        mock_response = {
            "label": "SFW",
            "is_nsfw": False,
            "nsfw_prob": 0.05,
            "sfw_prob": 0.95,
            "threshold": 0.7,
            "policy": "ALLOW",
            "confidence": 0.90,
            "confidence_level": "high",
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            res = await client.predict_image_url("https://example.com/cat.jpg", threshold=0.7)
            assert isinstance(res, ImageDetectionResult)
            assert res.is_nsfw is False
            assert res.label == "SFW"
            assert res.policy == "ALLOW"
            assert res.nsfw_prob == 0.05


@pytest.mark.asyncio
async def test_predict_image_url_451_nsfw(api_key):
    async with NSFWModerationAPI(api_key=api_key) as client:
        # Simulate HTTP 451 response from server
        error_body = '{"label": "NSFW", "nsfw_prob": 0.98, "policy": "BLOCK"}'
        client_error = ClientError("Unavailable For Legal Reasons", status_code=451, response_text=error_body)

        with patch("ayyad_apis.utils.BaseRapidAPI._make_request", side_effect=client_error):
            res = await client.predict_image_url("https://example.com/flagged.jpg")
            assert isinstance(res, ImageDetectionResult)
            assert res.is_nsfw is True
            assert res.label == "NSFW"
            assert res.policy == "BLOCK"


@pytest.mark.asyncio
async def test_predict_video_url(api_key):
    async with NSFWModerationAPI(api_key=api_key) as client:
        mock_response = {
            "nsfw": False,
            "reason": "safe",
            "policy": "ALLOW",
            "stats": {
                "max_prob": 0.12,
                "avg_prob": 0.04,
                "total_above_duration": 0.0,
            },
            "video": {
                "duration": 60.0,
                "sampled_duration": 60.0,
                "frames_checked": 60,
                "segments": [],
            },
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            res = await client.predict_video_url("https://example.com/movie.mp4")
            assert isinstance(res, VideoDetectionResult)
            assert res.nsfw is False
            assert res.stats.max_prob == 0.12
            assert res.video.duration == 60.0


@pytest.mark.asyncio
async def test_request_video_upload_url(api_key):
    async with NSFWModerationAPI(api_key=api_key) as client:
        mock_response = {
            "url": "https://upload.gateway.com/video?token=123",
            "expires_in": 86400,
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            res = await client.request_video_upload_url()
            assert isinstance(res, UploadUrl)
            assert res.url == "https://upload.gateway.com/video?token=123"
            assert res.expires_in == 86400


@pytest.mark.asyncio
async def test_upload_video_and_predict(api_key):
    async with NSFWModerationAPI(api_key=api_key) as client:
        expected_result = VideoDetectionResult(nsfw=False, success=True)
        with patch.object(client, "upload_video_and_analyze", new=AsyncMock(return_value=expected_result)) as mock_upload:
            res = await client.upload_video_and_predict("sample.mp4")
            mock_upload.assert_called_once_with("sample.mp4", config=None)
            assert res.nsfw is False


@pytest.mark.asyncio
async def test_is_safe_image_and_video(api_key, tmp_path):
    async with NSFWModerationAPI(api_key=api_key) as client:
        # Mock safe image URL
        with patch.object(client, "predict_image_url", new=AsyncMock(return_value=ImageDetectionResult(is_nsfw=False))):
            assert await client.is_safe("https://example.com/flower.jpg") is True

        # Mock unsafe image URL
        with patch.object(client, "predict_image_url", new=AsyncMock(return_value=ImageDetectionResult(is_nsfw=True))):
            assert await client.is_safe("https://example.com/nude.jpg") is False

        # Mock video URL
        with patch.object(client, "predict_video_url", new=AsyncMock(return_value=VideoDetectionResult(nsfw=False))):
            assert await client.is_safe("https://example.com/clip.mp4") is True

        # Mock local file
        test_file = tmp_path / "photo.png"
        test_file.write_bytes(b"dummy")
        with patch.object(client, "predict_image_upload", new=AsyncMock(return_value=ImageDetectionResult(is_nsfw=False))):
            assert await client.is_safe(test_file) is True


@pytest.mark.asyncio
async def test_scan_and_quarantine_directory(api_key, tmp_path):
    async with NSFWModerationAPI(api_key=api_key) as client:
        media_dir = tmp_path / "media"
        media_dir.mkdir()
        quarantine_dir = tmp_path / "quarantine"

        f_safe = media_dir / "safe.jpg"
        f_safe.write_bytes(b"safe_image")

        f_nsfw = media_dir / "bad.jpg"
        f_nsfw.write_bytes(b"bad_image")

        async def fake_predict_upload(file_path, threshold=0.70):
            if "bad" in str(file_path):
                return ImageDetectionResult(is_nsfw=True, label="NSFW", nsfw_prob=0.99)
            return ImageDetectionResult(is_nsfw=False, label="SFW", nsfw_prob=0.01)

        with patch.object(client, "predict_image_upload", side_effect=fake_predict_upload):
            # Test scan_directory
            scan_res = await client.scan_directory(media_dir)
            assert isinstance(scan_res, DirectoryScanResult)
            assert scan_res.total_files == 2
            assert len(scan_res.safe_files) == 1
            assert len(scan_res.flagged_files) == 1

            # Test quarantine_directory
            q_res = await client.quarantine_directory(media_dir, quarantine_dir=quarantine_dir)
            assert isinstance(q_res, QuarantineResult)
            assert len(q_res.quarantined_files) == 1
            assert not f_nsfw.exists()
            assert f_safe.exists()
            assert (quarantine_dir / "bad.jpg").exists()


@pytest.mark.asyncio
async def test_moderate_batch(api_key):
    async with NSFWModerationAPI(api_key=api_key) as client:
        with patch.object(client, "is_safe", new=AsyncMock(side_effect=[True, False])):
            res = await client.moderate_batch(["https://example.com/1.jpg", "https://example.com/2.jpg"])
            assert len(res) == 2
            assert res[0][1] is True
            assert res[1][1] is False
