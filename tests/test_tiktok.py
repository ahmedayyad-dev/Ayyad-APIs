import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ayyad_apis.tiktok import (
    TikTokAPI,
    TikTokDataAPI,
    TikTokResolvedLink,
    TikTokShareLink,
    TikTokVideoStatus,
    TikTokVideo,
    TikTokUserProfile,
    TikTokPostsResponse,
    TikTokMusicVideosResponse,
    TikTokRelatedMusicResponse,
    clean_tiktok_url,
    extract_aweme_id,
    TikTokUserAnalytics,
    TikTokBatchDownloadResult,
)
from ayyad_apis.utils import InvalidInputError


@pytest.fixture
def api_key():
    return "test_rapidapi_key"


@pytest.mark.asyncio
async def test_api_aliases_and_constants(api_key):
    assert TikTokDataAPI is TikTokAPI
    client = TikTokAPI(api_key=api_key)
    assert client.BASE_URL == "https://tiktok-data-api20.p.rapidapi.com"
    assert client.DEFAULT_HOST == "tiktok-data-api20.p.rapidapi.com"
    assert client.rapidapi_host == "tiktok-data-api20.p.rapidapi.com"


@pytest.mark.asyncio
async def test_resolve_link(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        mock_response = {
            "landing_url": "https://www.tiktok.com/@user/video/7705550414279675143",
            "status_code": 0,
            "status_msg": "",
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            res = await client.resolve_link("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/")
            mock_req.assert_called_once_with(
                "GET", "/resolve-link", params={"url": "https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/"}
            )
            assert isinstance(res, TikTokResolvedLink)
            assert res.landing_url == "https://www.tiktok.com/@user/video/7705550414279675143"
            assert res.aweme_id == "7705550414279675143"


@pytest.mark.asyncio
async def test_resolve_link_empty_validation(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        with pytest.raises(InvalidInputError):
            await client.resolve_link("")


@pytest.mark.asyncio
async def test_share_link(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        mock_response = {
            "shorten_url": "https://vm.tiktok.com/ZS9f5dvYQjQxw-qncWc/",
            "status_code": 0,
            "status_msg": "",
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            res = await client.share_link("https://www.tiktok.com/@tiktok/video/7705550414279675143")
            mock_req.assert_called_once_with(
                "GET", "/share-link", params={"url": "https://www.tiktok.com/@tiktok/video/7705550414279675143"}
            )
            assert isinstance(res, TikTokShareLink)
            assert res.shorten_url == "https://vm.tiktok.com/ZS9f5dvYQjQxw-qncWc/"


@pytest.mark.asyncio
async def test_video_status(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        mock_response = {
            "aweme_result": [
                {"aweme_id": "7705550414279675143", "pass_status_check": True},
                {"aweme_id": "123456", "pass_status_check": False},
            ],
            "status_code": 0,
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            res = await client.video_status(["7705550414279675143", "123456"])
            mock_req.assert_called_once_with(
                "GET", "/videos/status", params={"ids": "7705550414279675143,123456"}
            )
            assert len(res) == 2
            assert res[0].aweme_id == "7705550414279675143"
            assert res[0].pass_status_check is True
            assert res[1].pass_status_check is False


@pytest.mark.asyncio
async def test_get_video(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        mock_response = {
            "aweme_id": "7705550414279675143",
            "desc": "Check out this awesome video! #fyp",
            "create_time": 1794397252,
            "author": {
                "uid": "12345",
                "sec_uid": "MS4wLjABAAAA...",
                "unique_id": "testcreator",
                "nickname": "Test Creator",
                "custom_verify": "Verified badge",
            },
            "statistics": {
                "aweme_id": "7705550414279675143",
                "digg_count": 1000,
                "comment_count": 50,
                "play_count": 25000,
                "share_count": 120,
            },
            "music": {
                "id_str": "999888",
                "title": "Viral Sound",
                "author": "Sound Artist",
                "duration": 30,
                "play_url": {"url_list": ["https://cdn.tiktok.com/music.mp3"]},
            },
            "video": {
                "duration": 15000,
                "width": 1080,
                "height": 1920,
                "play_addr": {"url_list": ["https://cdn.tiktok.com/play.mp4"]},
                "download_addr": {"url_list": ["https://cdn.tiktok.com/download.mp4"]},
                "download_no_watermark_addr": ["https://cdn.tiktok.com/no_wm.mp4"],
            },
            "share_url": "https://www.tiktok.com/@testcreator/video/7705550414279675143",
            "region": "US",
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            video = await client.get_video("7705550414279675143")
            assert isinstance(video, TikTokVideo)
            assert video.aweme_id == "7705550414279675143"
            assert video.desc == "Check out this awesome video! #fyp"
            assert video.duration_seconds == 15.0
            assert video.duration_string == "0:15"
            assert video.author.unique_id == "testcreator"
            assert video.author.verified is True
            assert video.statistics.digg_count == 1000
            assert video.music.title == "Viral Sound"
            assert video.music.duration_string == "0:30"
            assert video.video.primary_video_url == "https://cdn.tiktok.com/no_wm.mp4"


@pytest.mark.asyncio
async def test_get_user(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        mock_response = {
            "user": {
                "uid": "12345",
                "sec_uid": "MS4wLjABAAAA...",
                "unique_id": "famous_user",
                "nickname": "Famous Star",
                "signature": "My official bio",
                "follower_count": 500000,
                "following_count": 120,
                "total_favorited": 10000000,
                "aweme_count": 250,
            }
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            profile = await client.get_user("MS4wLjABAAAA...")
            assert isinstance(profile, TikTokUserProfile)
            assert profile.unique_id == "famous_user"
            assert profile.follower_count == 500000
            assert profile.total_favorited == 10000000


@pytest.mark.asyncio
async def test_get_user_posts(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        mock_response = {
            "aweme_list": [
                {"aweme_id": "111", "desc": "Post 1"},
                {"aweme_id": "222", "desc": "Post 2"},
            ],
            "has_more": 1,
            "max_cursor": 123456789,
            "min_cursor": 0,
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_response)):
            posts = await client.get_user_posts("MS4wLjABAAAA...", count=2)
            assert isinstance(posts, TikTokPostsResponse)
            assert len(posts) == 2
            assert posts.has_more is True
            assert posts[0].aweme_id == "111"
            assert posts[1].aweme_id == "222"
            # Test iterable
            ids = [p.aweme_id for p in posts]
            assert ids == ["111", "222"]


@pytest.mark.asyncio
async def test_music_endpoints(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        mock_videos = {
            "aweme_list": [{"aweme_id": "999", "desc": "Dance to sound"}],
            "has_more": True,
            "cursor": 10,
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_videos)):
            res = await client.get_music_videos("sound123", count=1)
            assert isinstance(res, TikTokMusicVideosResponse)
            assert len(res) == 1
            assert res[0].aweme_id == "999"

        mock_related = {
            "music_list": [
                {"id_str": "sound456", "title": "Related Track", "author": "Other Artist"}
            ],
            "has_more": False,
        }
        with patch.object(client, "_make_request", new=AsyncMock(return_value=mock_related)):
            rel = await client.get_related_music("sound123")
            assert isinstance(rel, TikTokRelatedMusicResponse)
            assert len(rel) == 1
            assert rel[0].title == "Related Track"


@pytest.mark.asyncio
async def test_download_shortcuts(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        # Mock resolve_link and get_video
        mock_resolved = TikTokResolvedLink(
            landing_url="https://www.tiktok.com/@u/video/7705550414279675143"
        )
        mock_video = MagicMock(spec=TikTokVideo)
        mock_video.download_video = AsyncMock(return_value="video.mp4")
        mock_video.download_music = AsyncMock(return_value="audio.mp3")

        with patch.object(client, "resolve_link", new=AsyncMock(return_value=mock_resolved)), \
             patch.object(client, "get_video", new=AsyncMock(return_value=mock_video)):

            # Download by short URL
            res_v = await client.download_video("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/")
            assert res_v == "video.mp4"
            mock_video.download_video.assert_called_once()

            # Download music by short URL
            res_m = await client.download_music("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/")
            assert res_m == "audio.mp3"
            mock_video.download_music.assert_called_once()


def test_clean_tiktok_url():
    dirty = "https://www.tiktok.com/@user/video/7705550414279675143?_t=8rZ&is_from_webapp=1&sender_device=pc&foo=bar"
    cleaned = clean_tiktok_url(dirty)
    assert "_t" not in cleaned
    assert "sender_device" not in cleaned
    assert "foo=bar" in cleaned
    assert cleaned.startswith("https://www.tiktok.com/@user/video/7705550414279675143")


def test_extract_aweme_id():
    url = "https://www.tiktok.com/@creator/video/7705550414279675143?is_from_webapp=1"
    assert extract_aweme_id(url) == "7705550414279675143"
    assert extract_aweme_id("https://www.tiktok.com/@creator/photo/1234567890123456789") == "1234567890123456789"
    assert extract_aweme_id("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/") is None


@pytest.mark.asyncio
async def test_iter_user_posts(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        page1 = TikTokPostsResponse(
            aweme_list=[TikTokVideo(aweme_id="post1"), TikTokVideo(aweme_id="post2")],
            has_more=True,
            max_cursor=100,
        )
        page2 = TikTokPostsResponse(
            aweme_list=[TikTokVideo(aweme_id="post3")],
            has_more=False,
            max_cursor=200,
        )
        with patch.object(client, "get_user_posts", side_effect=[page1, page2]) as mock_posts:
            collected = []
            async for post in client.iter_user_posts("test_sec_uid", max_posts=3):
                collected.append(post.aweme_id)

            assert collected == ["post1", "post2", "post3"]
            assert mock_posts.call_count == 2


@pytest.mark.asyncio
async def test_iter_music_videos(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        page1 = TikTokMusicVideosResponse(
            aweme_list=[TikTokVideo(aweme_id="m1"), TikTokVideo(aweme_id="m2")],
            has_more=True,
            cursor=10,
        )
        page2 = TikTokMusicVideosResponse(
            aweme_list=[TikTokVideo(aweme_id="m3")],
            has_more=False,
            cursor=20,
        )
        with patch.object(client, "get_music_videos", side_effect=[page1, page2]) as mock_music:
            collected = []
            async for video in client.iter_music_videos("sound123", max_videos=3):
                collected.append(video.aweme_id)

            assert collected == ["m1", "m2", "m3"]
            assert mock_music.call_count == 2


@pytest.mark.asyncio
async def test_get_user_analytics(api_key):
    async with TikTokAPI(api_key=api_key) as client:
        from ayyad_apis.tiktok.core import TikTokStatistics
        v1 = TikTokVideo(aweme_id="v1", statistics=TikTokStatistics(play_count=1000, digg_count=100, comment_count=10, share_count=5))
        v2 = TikTokVideo(aweme_id="v2", statistics=TikTokStatistics(play_count=2000, digg_count=200, comment_count=20, share_count=10))

        async def fake_iter(sec_uid, max_posts=None):
            for v in [v1, v2]:
                yield v

        with patch.object(client, "iter_user_posts", side_effect=fake_iter):
            analytics = await client.get_user_analytics("sec123", sample_size=2)
            assert isinstance(analytics, TikTokUserAnalytics)
            assert analytics.identifier == "sec123"
            assert analytics.total_sampled_videos == 2
            assert analytics.total_views == 3000
            assert analytics.total_likes == 300
            assert analytics.avg_views == 1500.0
            assert analytics.avg_likes == 150.0
            assert analytics.avg_comments == 15.0
            assert analytics.avg_shares == 7.5
            # ((300 + 30 + 15) / 3000) * 100 = 345 / 30 = 11.5
            assert analytics.engagement_rate == 11.5


@pytest.mark.asyncio
async def test_download_user_videos(api_key, tmp_path):
    async with TikTokAPI(api_key=api_key) as client:
        v1 = TikTokVideo(aweme_id="v1", desc="Awesome day")
        v1.download_video = AsyncMock(return_value=str(tmp_path / "v1.mp4"))

        async def fake_iter(sec_uid, max_posts=None):
            yield v1

        with patch.object(client, "iter_user_posts", side_effect=fake_iter), \
             patch.object(client, "download_video", new=AsyncMock(return_value=str(tmp_path / "v1.mp4"))):
            batch_result = await client.download_user_videos("sec123", output_dir=tmp_path, max_videos=1)
            assert isinstance(batch_result, TikTokBatchDownloadResult)
            assert batch_result.total_requested == 1
            assert len(batch_result.downloaded) == 1
            assert len(batch_result.failed) == 0


@pytest.mark.asyncio
async def test_download_batch(api_key, tmp_path):
    async with TikTokAPI(api_key=api_key) as client:
        mock_resolved = TikTokResolvedLink(landing_url="https://www.tiktok.com/@u/video/123")
        with patch.object(client, "resolve_link", new=AsyncMock(return_value=mock_resolved)), \
             patch.object(client, "download_video", new=AsyncMock(side_effect=["/tmp/1.mp4", Exception("Network error")])):
            result = await client.download_batch(
                ["https://tiktok.com/@u/video/1", "https://tiktok.com/@u/video/2"],
                output_dir=tmp_path,
                concurrency=2,
            )
            assert isinstance(result, TikTokBatchDownloadResult)
            assert result.total_requested == 2
            assert len(result.downloaded) == 1
            assert len(result.failed) == 1
            assert result.failed[0][0] == "https://tiktok.com/@u/video/2"

