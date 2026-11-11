import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from ayyad_apis.tube_relay import (
    TubeRelayAPI,
    VideoInfo,
    StreamUrl,
    TubeRelayError,
)


@pytest.fixture
def api_key():
    return "test_api_key"


def test_validate_id():
    validate = TubeRelayAPI._validate_id
    # Bare ID
    assert validate("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Full URLs
    assert validate("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert validate("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert validate("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert validate("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    with pytest.raises(TubeRelayError):
        validate("")

    with pytest.raises(TubeRelayError):
        validate("short")


@pytest.mark.asyncio
async def test_info(api_key):
    async with TubeRelayAPI(api_key=api_key) as client:
        mock_response = {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "duration": 213,
            "duration_string": "03:33",
            "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            "channel": "Rick Astley",
            "views": 1500000000,
            "views_text": "1.5B views",
        }
        with patch.object(client, "_get_json", new=AsyncMock(return_value=mock_response)) as mock_req:
            info = await client.get_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            mock_req.assert_called_once_with("info", {"video_id": "dQw4w9WgXcQ"})
            assert isinstance(info, VideoInfo)
            assert info.id == "dQw4w9WgXcQ"
            assert info.title == "Rick Astley - Never Gonna Give You Up"
            assert info.duration == 213


@pytest.mark.asyncio
async def test_search(api_key):
    async with TubeRelayAPI(api_key=api_key) as client:
        mock_response = [
            {"id": "dQw4w9WgXcQ", "title": "Rick Astley", "duration": 213},
            {"id": "oHg5SJYRHA0", "title": "RickRoll'D", "duration": 212},
        ]
        with patch.object(client, "_get_json", new=AsyncMock(return_value=mock_response)) as mock_req:
            results = await client.search("rick astley", limit=2)
            mock_req.assert_called_once_with("search", {"query": "rick astley", "limit": 2})
            assert len(results) == 2
            assert results[0].id == "dQw4w9WgXcQ"
            assert results[1].id == "oHg5SJYRHA0"


def test_stream_url_generation(api_key):
    client = TubeRelayAPI(api_key=api_key)
    # TubeRelayAPI.stream constructs the direct streaming endpoint URL synchronously
    url = client.stream("https://www.youtube.com/watch?v=dQw4w9WgXcQ", type="audio", quality="best")
    assert "dQw4w9WgXcQ" in url
    assert "type=audio" in url
    assert "quality=best" in url


@pytest.mark.asyncio
async def test_download(api_key):
    async with TubeRelayAPI(api_key=api_key) as client:
        with patch("ayyad_apis.tube_relay.core._download_file", new=AsyncMock(return_value="song.mp3")) as mock_dl:
            res = await client.download("dQw4w9WgXcQ", file_path="song.mp3", type="audio")
            assert res == "song.mp3"
            mock_dl.assert_called_once()


@pytest.mark.asyncio
async def test_quick_download_with_url(api_key):
    async with TubeRelayAPI(api_key=api_key) as client:
        with patch.object(client, "download", new=AsyncMock(return_value="output.mp3")) as mock_dl:
            res = await client.quick_download(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                file_path="output.mp3",
                type="audio",
            )
            assert res == "output.mp3"
            mock_dl.assert_called_once_with(
                "dQw4w9WgXcQ",
                "output.mp3",
                type="audio",
                quality="best",
                max_retries=3,
            )


@pytest.mark.asyncio
async def test_quick_download_with_search(api_key):
    async with TubeRelayAPI(api_key=api_key) as client:
        mock_result = VideoInfo(id="dQw4w9WgXcQ", title="Rick Astley")
        with patch.object(client, "search", new=AsyncMock(return_value=[mock_result])), \
             patch.object(client, "download", new=AsyncMock(return_value="yt_dQw4w9WgXcQ.mp3")) as mock_dl:
            res = await client.quick_download("rick astley never gonna give you up")
            assert res == "yt_dQw4w9WgXcQ.mp3"
            mock_dl.assert_called_once_with(
                "dQw4w9WgXcQ",
                "yt_dQw4w9WgXcQ.mp3",
                type="audio",
                quality="best",
                max_retries=3,
            )
