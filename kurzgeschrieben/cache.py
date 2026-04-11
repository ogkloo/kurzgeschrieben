"""On-disk transcript cache.

Transcripts are essentially free to store and expensive to re-fetch (network
round-trip, possible IP rate-limits), and once we have one we also want to be
able to re-run the LLM layer against different summarization styles without
touching YouTube again. So the cache is treated as the authoritative source
for a given video id: if it's cached, we read from there and skip the network
entirely unless the caller asks for a fresh fetch.

Files live in `.cache/transcripts/{video_id}.json` relative to the user's
current working directory — the directory the tool was invoked from. Each
file carries a `schema` version so future shape changes can be rolled out
without crashing old caches; an unknown/mismatched schema is treated as a
cache miss, not an error.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .transcript import Snippet, Transcript, extract_video_id, fetch_transcript


# Bump this whenever the on-disk shape changes in a non-backwards-compatible
# way. Old files with a different schema are ignored (treated as a miss).
CACHE_SCHEMA_VERSION = 1


def default_cache_dir() -> Path:
    """Where the transcript cache lives by default.

    Resolved lazily (per call) rather than frozen at import time so the cache
    follows the user's working directory. Cloning the repo and running the
    tool from that clone puts the cache inside the clone, which is the
    ergonomic thing most of the time.
    """
    return Path.cwd() / ".cache" / "transcripts"


def load(video_id: str, cache_dir: Path | None = None) -> Transcript | None:
    """Return a cached `Transcript` for this video id, or None on miss.

    Any parse error, missing field, or schema mismatch is treated as a miss so
    the caller falls through to a fresh fetch. This is intentional: a stale or
    malformed cache file should never block the user from getting a transcript.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    path = _path_for(video_id, cache_dir)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(raw, dict) or raw.get("schema") != CACHE_SCHEMA_VERSION:
        return None

    try:
        snippets = tuple(
            Snippet(
                text=s["text"],
                start=float(s["start"]),
                duration=float(s["duration"]),
            )
            for s in raw["snippets"]
        )
        return Transcript(
            video_id=raw["video_id"],
            language=raw["language"],
            language_code=raw["language_code"],
            is_generated=bool(raw["is_generated"]),
            snippets=snippets,
        )
    except (KeyError, TypeError, ValueError):
        return None


def save(transcript: Transcript, cache_dir: Path | None = None) -> Path:
    """Write a transcript to the cache. Returns the path written."""
    if cache_dir is None:
        cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _path_for(transcript.video_id, cache_dir)

    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": transcript.video_id,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "snippets": [asdict(s) for s in transcript.snippets],
    }

    # Atomic write: dump to a sibling temp file and rename, so an interrupted
    # write can't leave a half-written JSON file in the cache.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path


def _path_for(video_id: str, cache_dir: Path) -> Path:
    # video_id is validated to [A-Za-z0-9_-]{11} in transcript.extract_video_id,
    # so it is safe to use directly as a filename.
    return cache_dir / f"{video_id}.json"


def get_transcript(
    url_or_id: str,
    *,
    languages: tuple[str, ...] = ("en",),
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> tuple[Transcript, bool]:
    """Cache-first transcript loader.

    Returns `(transcript, from_cache)`. If `refresh` is true or the cache has
    no entry for the video id, performs a network fetch and writes the result
    to the cache before returning. This is the function the CLI and the LLM
    layer should call — it makes the cache the default source of truth while
    still allowing a forced refetch.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()

    video_id = extract_video_id(url_or_id)

    if not refresh:
        cached = load(video_id, cache_dir=cache_dir)
        # Only accept a cache hit if the stored language actually satisfies
        # the caller's priority list — otherwise asking for --lang de after
        # having cached English would silently return the wrong language.
        if cached is not None and cached.language_code in languages:
            return cached, True

    fresh = fetch_transcript(video_id, languages=languages)
    save(fresh, cache_dir=cache_dir)
    return fresh, False
