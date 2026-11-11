import pytest
from ayyad_apis.youtube_to_telegram import YouTubeAPI


@pytest.fixture
def api_key():
    return "test_rapidapi_key"


def test_api_constants(api_key):
    client = YouTubeAPI(api_key=api_key)
    assert client.BASE_URL == "https://youtube-to-telegram-uploader-api.p.rapidapi.com"
    assert client.DEFAULT_HOST == "youtube-to-telegram-uploader-api.p.rapidapi.com"
    assert client.rapidapi_host == "youtube-to-telegram-uploader-api.p.rapidapi.com"
