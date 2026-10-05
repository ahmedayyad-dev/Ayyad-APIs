# Ayyad APIs

Async Python wrappers for three public APIs: YouTube metadata/download, NSFW
content detection, and YouTube streaming.

## Installation

```bash
pip install -U git+https://github.com/ahmedayyad-dev/Ayyad-APIs.git
```

## Clients

| Client | Base URL | Auth |
| --- | --- | --- |
| `YouTubeAPI` | `youtube-to-telegram-uploader-api.p.rapidapi.com` | RapidAPI key header |
| `PornDetectionAPI` | `porn-detection-api.p.rapidapi.com` | RapidAPI key header |
| `TubeRelayAPI` | `tuberelay.api.ahmedayyad.dev` | API key **in the URL path** |

All clients are asynchronous and support `async with`:

```python
from ayyad_apis import YouTubeAPI, PornDetectionAPI, TubeRelayAPI
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

### TubeRelayAPI

```python
async with TubeRelayAPI(api_key="...") as api:
    info = await api.get_info("dQw4w9WgXcQ")
    print(info.title, info.duration_string)

    for v in await api.search("python tutorial", limit=5):
        print(v.title)

    await api.download("dQw4w9WgXcQ", "song.m4a", type="audio")

    # Key-free URL that is safe to hand to a player (expires in 300s)
    public = await api.stream_url("dQw4w9WgXcQ", type="audio")
    print(public.url, public.expires_in)
```

`get_info` and `stream_url` require a bare 11–12 character video ID; passing a
full URL raises `TubeRelayError` immediately instead of returning an opaque 422.
Use `search()`, which accepts URLs, to resolve one first.

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

## Changelog

See [CHANGELOG.md](CHANGELOG.md). 0.3.0 contains breaking changes.