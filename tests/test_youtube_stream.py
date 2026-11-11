import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from ayyad_apis.youtube_stream import (
    YouTubeStreamAPI,
    YouTubeStreamMP3API,
    VideoInfo,
    StreamUrl,
    YouTubeStreamError,
)
from ayyad_apis.utils import InvalidInputError


@pytest.fixture
def api_key():
    return "test_rapidapi_key"


@pytest.mark.asyncio
async def test_api_aliases_and_constants(api_key):
    assert YouTubeStreamMP3API is YouTubeStreamAPI
    client = YouTubeStreamAPI(api_key=api_key)
    assert client.BASE_URL == "https://youtube-stream-mp3-url-api.p.rapidapi.com"
    assert client.DEFAULT_HOST == "youtube-stream-mp3-url-api.p.rapidapi.com"
    assert client.rapidapi_host == "youtube-stream-mp3-url-api.p.rapidapi.com"


def test_extract_video_id():
    extract = YouTubeStreamAPI._extract_video_id
    assert extract("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    with pytest.raises(InvalidInputError):
        extract("")


@pytest.mark.asyncio
async def test_get_info(api_key):
    async with YouTubeStreamAPI(api_key=api_key) as client:
        mock_response = {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "duration": 213,
            "duration_string": "03:33",
            "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            "channel": "Rick Astley",
            "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
            "views": 1500000000,
            "views_text": "1.5B views",
            "is_live": False,
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            info = await client.get_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            mock_req.assert_called_once_with("GET", "/info", params={"video_id": "dQw4w9WgXcQ"})
            assert isinstance(info, VideoInfo)
            assert info.id == "dQw4w9WgXcQ"
            assert info.title == "Rick Astley - Never Gonna Give You Up"
            assert info.duration == 213
            assert info.duration_string == "03:33"
            assert info._api is client


@pytest.mark.asyncio
async def test_search(api_key):
    async with YouTubeStreamAPI(api_key=api_key) as client:
        mock_response = [
            {
                "id": "dQw4w9WgXcQ",
                "title": "Rick Astley",
                "duration": 213,
                "duration_string": "03:33",
            },
            {
                "id": "oHg5SJYRHA0",
                "title": "RickRoll'D",
                "duration": 212,
                "duration_string": "03:32",
            },
        ]
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            results = await client.search("rick astley", limit=2)
            mock_req.assert_called_once_with("GET", "/search", params={"query": "rick astley", "limit": 2})
            assert len(results) == 2
            assert results[0].id == "dQw4w9WgXcQ"
            assert results[1].id == "oHg5SJYRHA0"


@pytest.mark.asyncio
async def test_stream_url(api_key):
    async with YouTubeStreamAPI(api_key=api_key) as client:
        mock_response = {
            "url": "https://tuberelay.api.ahmedayyad.dev/stream/123",
            "expires_in": 300,
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            res = await client.stream_url("dQw4w9WgXcQ", type="audio", quality="best")
            mock_req.assert_called_once_with(
                "GET", "/stream-url", params={"video_id": "dQw4w9WgXcQ", "type": "audio", "quality": "best"}
            )
            assert isinstance(res, StreamUrl)
            assert res.url == "https://tuberelay.api.ahmedayyad.dev/stream/123"
            assert res.expires_in == 300


@pytest.mark.asyncio
async def test_stream_url_invalid_type(api_key):
    async with YouTubeStreamAPI(api_key=api_key) as client:
        with pytest.raises(InvalidInputError):
            await client.stream_url("dQw4w9WgXcQ", type="invalid_type")


@pytest.mark.asyncio
async def test_download(api_key):
    async with YouTubeStreamAPI(api_key=api_key) as client:
        mock_stream = StreamUrl(url="https://cdn.example.com/stream.mp3", expires_in=300)
        with patch.object(client, "stream_url", new=AsyncMock(return_value=mock_stream)), \
             patch("ayyad_apis.youtube_stream.core._download_file", new=AsyncMock(return_value="song.mp3")) as mock_dl:
            res = await client.download("dQw4w9WgXcQ", file_path="song.mp3", type="audio")
            assert res == "song.mp3"
            mock_dl.assert_called_once()


@pytest.mark.asyncio
async def test_quick_download(api_key):
    async with YouTubeStreamAPI(api_key=api_key) as client:
        mock_search_item = VideoInfo(id="dQw4w9WgXcQ", title="Rick Astley")
        with patch.object(client, "search", new=AsyncMock(return_value=[mock_search_item])), \
             patch.object(client, "download", new=AsyncMock(return_value="downloaded.mp3")) as mock_dl:
            res = await client.quick_download("rick astley never gonna give you up", output_path="downloaded.mp3")
            assert res == "downloaded.mp3"
            mock_dl.assert_called_once_with(
                "dQw4w9WgXcQ",
                file_path="downloaded.mp3",
                type="audio",
                quality="best",
                show_progress=False,
                progress_callback=None,
            )
