"""On-disk summary cache.

Mirrors the shape of `cache.py` (the transcript cache) but for LLM outputs:
we store the generated article itself so re-reading a video is free and we
can power a "previously summarized" list on the web app's index page from
disk alone.

Each summary is keyed by `(video_id, style)`, stored as
`.cache/summaries/{video_id}__{style}.json` relative to the user's current
working directory. The shape is versioned so a future prompt-format or field
change can be rolled out without crashing old entries — an unknown schema is
treated as a cache miss.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


# Bump whenever the on-disk shape changes incompatibly.
CACHE_SCHEMA_VERSION = 1


def default_cache_dir() -> Path:
    """Where the summary cache lives by default. Resolved per call so the
    cache follows the user's working directory rather than the install path."""
    return Path.cwd() / ".cache" / "summaries"


@dataclass(frozen=True)
class Summary:
    """A cached LLM-generated summary of a video."""

    video_id: str
    video_title: str
    video_author: str | None
    style: str
    model: str
    content: str  # markdown
    created_at: str  # ISO 8601 UTC


def load(
    video_id: str,
    style: str,
    cache_dir: Path | None = None,
) -> Summary | None:
    """Return a cached summary for (video_id, style) or None on miss.

    Any parse error, missing field, or schema mismatch is treated as a miss so
    a stale/corrupt file never blocks the user from regenerating.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    path = _path_for(video_id, style, cache_dir)
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
        return Summary(
            video_id=raw["video_id"],
            video_title=raw["video_title"],
            video_author=raw.get("video_author"),
            style=raw["style"],
            model=raw["model"],
            content=raw["content"],
            created_at=raw["created_at"],
        )
    except (KeyError, TypeError):
        return None


def save(summary: Summary, cache_dir: Path | None = None) -> Path:
    """Write a summary to the cache atomically. Returns the written path."""
    if cache_dir is None:
        cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _path_for(summary.video_id, summary.style, cache_dir)

    payload = {"schema": CACHE_SCHEMA_VERSION, **asdict(summary)}

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path


def delete(
    video_id: str,
    style: str,
    cache_dir: Path | None = None,
) -> bool:
    """Remove a cached summary. Returns True if a file was removed.

    Used by the regenerate path — deleting the summary file but keeping the
    transcript cache means re-running is a single LLM call away, with no
    YouTube round-trip.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    path = _path_for(video_id, style, cache_dir)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def list_all(cache_dir: Path | None = None) -> list[Summary]:
    """Return every cached summary, sorted newest-first by created_at.

    Malformed files are silently skipped so one bad entry can't take down the
    index page.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    if not cache_dir.exists():
        return []

    summaries: list[Summary] = []
    for path in cache_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(raw, dict) or raw.get("schema") != CACHE_SCHEMA_VERSION:
            continue

        try:
            summaries.append(
                Summary(
                    video_id=raw["video_id"],
                    video_title=raw["video_title"],
                    video_author=raw.get("video_author"),
                    style=raw["style"],
                    model=raw["model"],
                    content=raw["content"],
                    created_at=raw["created_at"],
                )
            )
        except (KeyError, TypeError):
            continue

    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def now_iso() -> str:
    """UTC ISO-8601 timestamp with seconds resolution — the canonical form used
    in `created_at` so string-sort == chronological sort."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path_for(video_id: str, style: str, cache_dir: Path) -> Path:
    # video_id is validated to [A-Za-z0-9_-]{11} upstream, and style is drawn
    # from a fixed enum of prompt names, so both are safe as filename parts.
    return cache_dir / f"{video_id}__{style}.json"
