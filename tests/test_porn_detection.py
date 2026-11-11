import pytest
from unittest.mock import AsyncMock, patch

from ayyad_apis.porn_detection import (
    PornDetectionAPI,
    ImageDetectionResult,
    VideoDetectionResult,
    UploadUrl,
)


@pytest.fixture
def api_key():
    return "test_rapidapi_key"


@pytest.mark.asyncio
async def test_api_constants(api_key):
    client = PornDetectionAPI(api_key=api_key)
    assert client.BASE_URL == "https://porn-detection-api.p.rapidapi.com"
    assert client.DEFAULT_HOST == "porn-detection-api.p.rapidapi.com"
    assert client.rapidapi_host == "porn-detection-api.p.rapidapi.com"


@pytest.mark.asyncio
async def test_predict_image_url(api_key):
    async with PornDetectionAPI(api_key=api_key) as client:
        mock_response = {
            "label": "SFW",
            "is_nsfw": False,
            "nsfw_prob": 0.05,
            "sfw_prob": 0.95,
            "threshold": 0.7,
            "policy": "ALLOW",
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            res = await client.predict_image_url("https://example.com/safe.jpg")
            assert isinstance(res, ImageDetectionResult)
            assert res.is_nsfw is False
            assert res.label == "SFW"


@pytest.mark.asyncio
async def test_predict_video_url(api_key):
    async with PornDetectionAPI(api_key=api_key) as client:
        mock_response = {
            "nsfw": False,
            "reason": "safe",
            "stats": {"max_prob": 0.1},
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            res = await client.predict_video_url("https://example.com/video.mp4")
            assert isinstance(res, VideoDetectionResult)
            assert res.nsfw is False
