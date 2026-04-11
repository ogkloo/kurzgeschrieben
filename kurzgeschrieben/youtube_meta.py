"""YouTube video metadata via the public oEmbed endpoint.

We only need the video *title* (and optionally the channel name) for display
in the web UI — the summary itself comes from the transcript. YouTube exposes
an oEmbed endpoint at `https://www.youtube.com/oembed` that returns title and
author without requiring an API key, which is exactly the right fit for a
local-only tool that doesn't want to carry YouTube Data API credentials.

Failures here must never block a summary from being created or displayed:
if the title can't be fetched we fall back to the bare video id so the page
is still usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


OEMBED_URL = "https://www.youtube.com/oembed"
OEMBED_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    title: str
    author: str | None


def fetch_meta(video_id: str) -> VideoMeta:
    """Fetch `(title, author)` for a video id, falling back to the id.

    Never raises on network/HTTP error — returns a `VideoMeta` whose `title`
    is the bare video id when the remote call fails. The caller should treat
    this as "best effort" metadata.
    """
    try:
        response = requests.get(
            OEMBED_URL,
            params={
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
            },
            timeout=OEMBED_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return VideoMeta(video_id=video_id, title=video_id, author=None)

    title = data.get("title") or video_id
    author = data.get("author_name") or None
    return VideoMeta(video_id=video_id, title=title, author=author)
