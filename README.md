# Ayyad APIs

Async Python wrappers for public APIs: YouTube metadata/download, NSFW
content detection and moderation, YouTube streaming, Anghami catalog search/streaming,
and TikTok public data extraction.

## Installation

```bash
pip install -U git+https://github.com/ahmedayyad-dev/Ayyad-APIs.git
```

## Clients

| Client | Base URL | Auth |
| --- | --- | --- |
| `YouTubeAPI` | `youtube-to-telegram-uploader-api.p.rapidapi.com` | RapidAPI key header |
| `YouTubeStreamAPI` | `youtube-stream-mp3-url-api.p.rapidapi.com` | RapidAPI key header |
| `PornDetectionAPI` | `porn-detection-api.p.rapidapi.com` | RapidAPI key header |
| `NSFWModerationAPI` | `nsfw-image-video-moderation-api.p.rapidapi.com` | RapidAPI key header |
| `TubeRelayAPI` | `tuberelay.api.ahmedayyad.dev` | API key **in the URL path** |
| `AnghamiAPI` | `anghami-music-data-and-stream.p.rapidapi.com` | RapidAPI key header |
| `TikTokAPI` | `tiktok-data-api20.p.rapidapi.com` | RapidAPI key header |

All clients are asynchronous and support `async with`:

```python
from ayyad_apis import (
    YouTubeAPI,
    YouTubeStreamAPI,
    PornDetectionAPI,
    NSFWModerationAPI,
    TubeRelayAPI,
    AnghamiAPI,
    TikTokAPI,
)
```

### YouTubeAPI

```python
async with YouTubeAPI(api_key="...") as api:
    info = await api.video_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(info.title, info.duration_string)

    results = await api.search("never gonna give you up", limit=5)
    for r in results:
        print(r.title, r.duration_string)

    sent = await api.youtube_to_telegram("dQw4w9WgXcQ", file_format="m4a")
    print(sent.file_url)

    # Upload to Google Photos (returns permanent CDN URL and direct audio stream for m4a/mp3)
    google = await api.youtube_to_google("dQw4w9WgXcQ", file_format="m4a")
    print(google.video_url, google.audio_url)
    # Automatically strips the 1-frame thumbnail video and converts to pure audio
    await google.download_audio("song.mp3")

    server = await api.youtube_to_server("dQw4w9WgXcQ", file_format="mp4")
    local = await server.download_file("video.mp4")

    live = await api.youtube_live_hls("dQw4w9WgXcQ")
    print(live.url)
```

If the same video is already being processed, the API answers HTTP 202. Catch it
and retry later — there is no progress endpoint to poll:

```python
from ayyad_apis import DownloadInProgressError

try:
    sent = await api.youtube_to_telegram("dQw4w9WgXcQ")
except DownloadInProgressError as e:
    print("already running:", e.response.job_id)
```

### PornDetectionAPI

`predict_*` endpoints answer HTTP 451 when content is flagged; that is treated as
a result, so you always get a populated object back.

```python
async with PornDetectionAPI(api_key="...") as api:
    image = await api.predict_image_url("https://example.com/photo.jpg")
    print(image.label, image.is_nsfw, image.policy)

    video = await api.predict_video_url("https://example.com/clip.mp4")
    print(video.nsfw, video.reason)
    print(video.stats.max_prob, video.video.duration_formatted)
    for seg in video.video.segments:
        print(f"  {seg.start}-{seg.end}s")

    # Full analysis: per-class scores + bounding box, never 451
    detail = await api.analyze_video("https://example.com/clip.mp4")
    print(detail.bbox, detail.worst_frame)
```

Local files:

```python
    await api.predict_image_upload("photo.jpg")
    result = await api.upload_video_and_analyze("clip.mp4")
```

> `upload_video_and_analyze` relies on the API's self-reported upload URL, which
> it builds from its own internal base URL. It only works when the API is fronted
> by a gateway that injects the required proxy-secret header.

### NSFWModerationAPI

Async client for the NSFW Image & Video Moderation API (`nsfw-image-video-moderation-api.p.rapidapi.com`).

```python
async with NSFWModerationAPI(api_key="...") as api:
    # URL prediction (safe responses or HTTP 451 non-error handling)
    image = await api.predict_image_url("https://example.com/photo.jpg", threshold=0.7)
    print(image.label, image.is_nsfw, image.scores)

    # Local file upload
    result = await api.predict_image_upload("photo.jpg")
    print(result.label, result.policy)

    # Video moderation via URL
    video = await api.predict_video_url("https://example.com/video.mp4")
    print(video.nsfw, video.reason)

    # Video upload & moderation via temporary presigned upload URL
    video_res = await api.upload_video_and_predict("clip.mp4")
    print(video_res.nsfw, video_res.stats.max_prob)

    # Developer Tools:
    # 1. Quick boolean safety check (transparently accepts URL or local path, image or video)
    if await api.is_safe("https://example.com/photo.jpg"):
        print("Safe to publish!")

    # 2. Concurrently audit a local directory of images/videos
    scan = await api.scan_directory("uploads/", recursive=True)
    print(f"Total: {scan.total_files}, Safe: {len(scan.safe_files)}, Flagged: {len(scan.flagged_files)}")

    # 3. Automatically quarantine flagged files to a separate directory
    quarantine = await api.quarantine_directory("uploads/", quarantine_dir="quarantined/")
    print("Quarantined files:", quarantine.quarantined_files)

    # 4. Batch moderate multiple targets concurrently
    results = await api.moderate_batch(["https://example.com/1.jpg", "photo.png"])
    for target, is_safe, err in results:
        print(target, "Safe:", is_safe)
```

### TubeRelayAPI

```python
async with TubeRelayAPI(api_key="...") as api:
    # Accepts bare 11-char ID or full YouTube URL (e.g., https://youtu.be/... or watch?v=...)
    info = await api.get_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(info.title, info.duration_string)

    for v in await api.search("python tutorial", limit=5):
        print(v.title)

    await api.download("dQw4w9WgXcQ", "song.m4a", type="audio")

    # Key-free URL that is safe to hand to a player (expires in 300s)
    public = await api.stream_url("dQw4w9WgXcQ", type="audio")
    print(public.url, public.expires_in)

    # One-liner search & download:
    await api.quick_download("rick astley never gonna give you up", "song.mp3")
```

`get_info`, `stream_url`, and `download` automatically validate and extract the 11-character video ID from bare IDs or any valid YouTube URL (including `youtu.be`, `watch?v=`, and `/shorts/`).

### YouTubeStreamAPI

Async client for the YouTube Stream MP3 URL API hosted on RapidAPI
(`youtube-stream-mp3-url-api.p.rapidapi.com`).

Unlike `TubeRelayAPI`, this client authenticates with standard RapidAPI key headers
and automatically accepts full YouTube URLs (e.g. `youtu.be`, `watch?v=...`, `shorts/...`)
without forcing manual ID extraction:

```python
async with YouTubeStreamAPI(api_key="...") as api:
    # Accepts URL or 11-char ID directly
    info = await api.get_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(info.title, info.duration_string)

    # Search YouTube
    results = await api.search("lo-fi hip hop", limit=5)
    for r in results:
        print(r.id, r.title)

    # Temporary CDN stream URL (expires in 300s)
    stream = await api.stream_url(info.id, type="audio", quality="best")
    print(stream.url)

    # Download audio or video directly
    await api.download(info.id, "rick.mp3", type="audio")

    # One-liner search & download:
    await api.quick_download("rick astley never gonna give you up", "song.mp3")
```

### AnghamiAPI

```python
async with AnghamiAPI(api_key="...") as api:
    found = await api.search("Amr Diab")
    print(found.query, found.count)

    for song in found:
        print(song.id, song.title, song.artist, song.duration_string)
        print(song.cover_art_url())

    detail = await api.get_song(found[0].id)
    print(detail.genre, detail.is_explicit)

    stream = await api.stream_url(detail.id, quality="128")
    print(stream.url)

    await api.download(detail.id, "song.m4a", quality="128")

    # Developer Tools:
    # 1. One-liner search & download:
    await api.quick_download("Amr Diab Tamally Maak", output_path="tamally.m4a")

    # 2. Related track recommendations:
    recommendations = await api.get_recommendations(detail.id)
    for rec in recommendations:
        print(rec.id, rec.title, rec.artist)
```

`search()` returns a `SearchResponse`, which is iterable and indexable, so it can
be used directly in a `for` loop. `filter_type` defaults to `"top"` and is
passed through to Anghami.

`stream_url()` resolves a time-limited, key-free CDN URL. Resolve it immediately
before playing or downloading; it expires. `download()` does that for you.

`quality` accepts `"24"`, `"64"`, `"128"`, or `"196"` (default `"64"`). Note it
is **196**, not 192.

### TikTokAPI

Async client for the TikTok Data API (`tiktok-data-api20.p.rapidapi.com`).
Read-only access to public TikTok data without login or cookies: link resolution,
video details with CDN streams, user profiles, posts, and sound discovery.

```python
async with TikTokAPI(api_key="...") as api:
    # Resolve shortened link (e.g. vm.tiktok.com)
    resolved = await api.resolve_link("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/")
    print(resolved.landing_url, resolved.aweme_id)

    # Check video availability status
    statuses = await api.video_status(["7705550414279675143"])
    print(statuses[0].aweme_id, statuses[0].pass_status_check)

    # Get full video metadata & statistics
    video = await api.get_video("7705550414279675143")
    print(video.desc, video.statistics.digg_count, video.statistics.play_count)
    print("Direct CDN URL:", video.video.primary_video_url)

    # Download video or audio stream
    await video.download_video("video.mp4")
    await video.download_music("audio.mp3")

    # Or 1-line direct download from URL or ID:
    await api.download_video("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/", "video.mp4")
    await api.download_music("https://vm.tiktok.com/ZS9fBknLstd2w-34rzR/", "audio.mp3")

    # Author profile
    profile = await api.get_user(video.author.sec_uid)
    print(profile.nickname, profile.follower_count, profile.following_count)

    # Author posts (paginated, iterable)
    posts = await api.get_user_posts(profile.sec_uid, count=20)
    for p in posts:
        print(p.aweme_id, p.desc, p.duration_string)

    # Videos by sound & related sounds
    sound_videos = await api.get_music_videos("7705550640109390598", count=10)
    related = await api.get_related_music("7705550640109390598")

    # Developer Tools:
    # 1. Async Generator Paginator (seamlessly stream across pages):
    async for post in api.iter_user_posts(profile.sec_uid, max_posts=50):
        print(post.aweme_id, post.desc)

    async for video in api.iter_music_videos("7705550640109390598", max_videos=30):
        print(video.aweme_id)

    # 2. Creator Analytics Engine:
    stats = await api.get_user_analytics(profile.sec_uid, sample_size=20)
    print(f"Engagement Rate: {stats.engagement_rate}%")
    print(f"Avg Views: {stats.avg_views}, Avg Likes: {stats.avg_likes}")
    if stats.top_video:
        print("Best Video:", stats.top_video.desc)

    # 3. Bulk & Batch Downloads (with concurrency and progress tracking):
    batch = await api.download_user_videos(
        profile.sec_uid,
        output_dir="downloads/creator/",
        max_videos=10,
        concurrency=3,
        progress_callback=lambda file, done, total: print(f"[{done}/{total}] {file}")
    )
    print("Downloaded:", len(batch.downloaded), "Skipped:", len(batch.skipped))

    # Or batch download a list of arbitrary URLs or IDs:
    await api.download_batch(["https://vm.tiktok.com/...", "7705550414279675143"], output_dir="batch/")
```

## Shared behaviour

### Sessions

Clients share one global `aiohttp` session, which is bound to the event loop that
created it. If you need to use a client across more than one event loop, pass
`own_session=True` or call `close_session()`:

```python
async with YouTubeAPI(api_key="...", own_session=True) as api:
    ...
```

### Errors

```
APIError
├── AuthenticationError   401/403
├── ClientError           other 4xx — never retried
├── InvalidInputError     bad argument or missing local file
├── DownloadError         a download could not be completed
└── RequestError          5xx / network — retried with backoff
    ├── APIResponseError
    ├── DownloadInProgressError
    └── TubeRelayError
```

`with_retry` retries 5xx and network failures with exponential backoff. 4xx
errors and HTTP 202 are raised immediately, since retrying cannot fix them. Set
`non_retryable=True` on any `APIError`, or `non_retryable = True` on a subclass,
to opt out explicitly.

`UploadError` is an alias of `RequestError`, kept for backwards compatibility.

### Downloads

`download_file` retries on failure, removes partial files, and treats a zero-byte
result as a failure — the streaming endpoints answer HTTP 200 with an empty body
when every upstream path fails.

## Notes and caveats

- **`TubeRelay` search `views` is approximate.** It is parsed from YouTube's
  abbreviated text, so `"1.2M views"` becomes `12`. Use `get_info` for an exact
  count.
- **`TubeRelay` streams carry no metadata.** No `Content-Type`, filename or
  `Content-Length`, so you choose the extension. Audio may arrive as raw
  AAC/ADTS rather than an m4a container.
- **`TubeRelay` ignores redirect-following on purpose.** An invalid key returns
  HTTP 307 to the login page; this is converted into a clear `TubeRelayError`.
- **`TubeRelay` search has two response shapes.** `published` only appears on
  uncached results, while `description` and `keywords` only appear on results
  served from the metadata cache.
- **`ServerResponse.key`** is derived from `download_url`; the API does not send
  it as a top-level field.
- **`youtube_to_server` URLs expire after 10 minutes.**