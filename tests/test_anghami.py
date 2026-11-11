import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from ayyad_apis.anghami import (
    AnghamiAPI,
    SongSummary,
    SongDetail,
    StreamUrl,
    SearchResponse,
)
from ayyad_apis.utils import InvalidInputError, RequestError


@pytest.fixture
def api_key():
    return "test_rapidapi_key"


@pytest.mark.asyncio
async def test_api_constants(api_key):
    client = AnghamiAPI(api_key=api_key)
    assert client.BASE_URL == "https://anghami-music-data-and-stream.p.rapidapi.com"
    assert client.DEFAULT_HOST == "anghami-music-data-and-stream.p.rapidapi.com"
    assert client.rapidapi_host == "anghami-music-data-and-stream.p.rapidapi.com"


@pytest.mark.asyncio
async def test_search(api_key):
    async with AnghamiAPI(api_key=api_key) as client:
        mock_response = {
            "status": "ok",
            "results": [
                {
                    "id": "1001",
                    "title": "Tamally Maak",
                    "artist": "Amr Diab",
                    "duration": 269,
                    "coverArt": 12345,
                    "genre": "Arabic Pop",
                }
            ],
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            results = await client.search("Amr Diab")
            mock_req.assert_called_once_with("GET", "/search", params={"q": "Amr Diab", "filter_type": "top"})
            assert isinstance(results, SearchResponse)
            assert len(results.results) == 1
            song = results.results[0]
            assert song.id == "1001"
            assert song.title == "Tamally Maak"
            assert song.artist == "Amr Diab"
            assert song.duration_string == "4:29"
            assert "12345" in song.cover_art_url()


@pytest.mark.asyncio
async def test_get_song(api_key):
    async with AnghamiAPI(api_key=api_key) as client:
        mock_response = {
            "id": "1001",
            "title": "Tamally Maak",
            "artist": "Amr Diab",
            "duration": 269,
            "sections": [
                {
                    "type": "recommendations",
                    "data": [
                        {"id": "1002", "title": "Nour El Ain", "artist": "Amr Diab"},
                        {"id": "1003", "title": "Amarain", "artist": "Amr Diab"},
                    ],
                }
            ],
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            detail = await client.get_song("1001")
            assert isinstance(detail, SongDetail)
            assert detail.id == "1001"
            assert detail.title == "Tamally Maak"
            assert len(detail.sections) == 1


@pytest.mark.asyncio
async def test_get_recommendations(api_key):
    async with AnghamiAPI(api_key=api_key) as client:
        mock_detail = SongDetail(
            id="1001",
            title="Tamally Maak",
            sections=[
                {
                    "data": [
                        {"id": "1002", "title": "Nour El Ain", "artist": "Amr Diab"},
                        {"id": "1003", "title": "Amarain", "artist": "Amr Diab"},
                    ]
                }
            ],
        )
        with patch.object(client, "get_song", new=AsyncMock(return_value=mock_detail)):
            recs = await client.get_recommendations("1001")
            assert len(recs) == 2
            assert recs[0].id == "1002"
            assert recs[0].title == "Nour El Ain"
            assert recs[1].id == "1003"
            assert recs[1].title == "Amarain"


@pytest.mark.asyncio
async def test_stream_url(api_key):
    async with AnghamiAPI(api_key=api_key) as client:
        mock_response = {
            "song_id": "1001",
            "quality": "128",
            "url": "https://cdn.anghami.com/stream/track.m4a",
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            stream = await client.stream_url("1001", quality="128")
            assert isinstance(stream, StreamUrl)
            assert stream.song_id == "1001"
            assert stream.quality == "128"
            assert stream.url == "https://cdn.anghami.com/stream/track.m4a"


@pytest.mark.asyncio
async def test_quick_download(api_key):
    async with AnghamiAPI(api_key=api_key) as client:
        mock_search_res = SearchResponse(
            query="tamally maak",
            results=[SongSummary(id="1001", title="Tamally Maak", artist="Amr Diab")],
        )
        with patch.object(client, "search", new=AsyncMock(return_value=mock_search_res)), \
             patch.object(client, "download", new=AsyncMock(return_value="Amr Diab - Tamally Maak.m4a")) as mock_dl:
            saved = await client.quick_download("tamally maak")
            assert saved == "Amr Diab - Tamally Maak.m4a"
            mock_dl.assert_called_once_with(
                "1001",
                file_path="Amr Diab - Tamally Maak.m4a",
                quality="128",
                max_retries=3,
            )
