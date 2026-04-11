"""YouTube transcript fetching.

Accepts any common YouTube URL form (or a bare 11-char video id), resolves it
to a video id, and returns a normalized `Transcript` with both timestamped
snippets and a joined plain-text form.

Prefers manually-created captions over auto-generated ones when both exist,
since auto-generated captions are typically noisier and worse for downstream
LLM summarization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)


DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)


@dataclass(frozen=True)
class Snippet:
    """A single timestamped transcript segment."""

    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class Transcript:
    """A fetched transcript in a provider-agnostic shape."""

    video_id: str
    language: str
    language_code: str
    is_generated: bool
    snippets: tuple[Snippet, ...]

    @property
    def plain_text(self) -> str:
        """Snippets joined into a single block of text.

        YouTube snippets are often split mid-sentence and individually stripped,
        so a single space join is the closest thing to "what was said" without
        trying to guess sentence boundaries — that's the LLM's job.
        """
        return " ".join(s.text.strip() for s in self.snippets if s.text.strip())

    @property
    def duration_seconds(self) -> float:
        """Approximate video length in seconds, derived from the last snippet.

        This is the transcript's coverage, not necessarily the video's true
        length — silent leading/trailing sections may be omitted by YouTube —
        but it's the right signal for "about how long is this video" framing
        and it requires no extra network calls.
        """
        if not self.snippets:
            return 0.0
        last = self.snippets[-1]
        return last.start + last.duration

    def timestamped_text(self) -> str:
        """Snippets joined as `[M:SS] text` lines, one per snippet.

        Preserves the timing information `plain_text` throws away. The LLM
        uses these timestamps to detect the video's natural segment boundaries
        so the generated article's section structure can follow the video's
        own pacing rather than an invented abstract taxonomy.

        For videos past one hour, the minute component rolls past 60 (e.g.
        `[67:42]`). A fixed `M:SS` format is simpler than a conditional
        `H:MM:SS` and still unambiguous for the model.
        """
        lines = []
        for s in self.snippets:
            text = s.text.strip()
            if not text:
                continue
            minutes = int(s.start) // 60
            seconds = int(s.start) % 60
            lines.append(f"[{minutes}:{seconds:02d}] {text}")
        return "\n".join(lines)


class TranscriptError(Exception):
    """Raised when a transcript cannot be fetched for a video."""


def extract_video_id(url_or_id: str) -> str:
    """Resolve any common YouTube URL form (or a bare id) to an 11-char video id.

    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID (and &t=, &list=, etc.)
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/live/VIDEO_ID
      - Bare VIDEO_ID (11 chars)
    """
    value = url_or_id.strip()
    if not value:
        raise TranscriptError("empty video id / url")

    # Bare id: YouTube ids are 11 chars from [A-Za-z0-9_-].
    if _looks_like_video_id(value):
        return value

    parsed = urlparse(value)
    host = (parsed.netloc or "").lower().removeprefix("www.")

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/", 1)[0]
        if _looks_like_video_id(candidate):
            return candidate

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            vs = parse_qs(parsed.query).get("v", [])
            if vs and _looks_like_video_id(vs[0]):
                return vs[0]

        # /shorts/ID, /embed/ID, /live/ID, /v/ID
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            if _looks_like_video_id(parts[1]):
                return parts[1]

    raise TranscriptError(f"could not extract a YouTube video id from: {url_or_id!r}")


def _looks_like_video_id(value: str) -> bool:
    if len(value) != 11:
        return False
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    )
    return all(ch in allowed for ch in value)


def fetch_transcript(
    url_or_id: str,
    languages: Sequence[str] | Iterable[str] = DEFAULT_LANGUAGES,
) -> Transcript:
    """Fetch a transcript for the given YouTube URL or video id.

    Strategy:
      1. Resolve the video id.
      2. List available transcripts for the video.
      3. Prefer a manually-created transcript in the requested languages.
      4. Fall back to an auto-generated transcript in the requested languages.
      5. Raise `TranscriptError` if neither exists.

    `languages` is an ordered priority list (first match wins), matching the
    upstream `youtube_transcript_api` convention.
    """
    langs = tuple(languages)
    if not langs:
        langs = DEFAULT_LANGUAGES

    video_id = extract_video_id(url_or_id)

    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
    except TranscriptsDisabled as err:
        raise TranscriptError(
            f"transcripts are disabled for video {video_id}"
        ) from err
    except YouTubeTranscriptApiException as err:
        # Covers VideoUnavailable, InvalidVideoId, IpBlocked, RequestBlocked,
        # PoTokenRequired, AgeRestricted, etc. — surface as one error type so
        # callers don't need to know the upstream class hierarchy.
        raise TranscriptError(
            f"could not fetch transcripts for video {video_id}: {err.__class__.__name__}"
        ) from err

    # Prefer human captions; fall back to auto-generated.
    try:
        transcript = transcript_list.find_manually_created_transcript(langs)
    except NoTranscriptFound:
        try:
            transcript = transcript_list.find_generated_transcript(langs)
        except NoTranscriptFound as err:
            available = ", ".join(
                f"{t.language_code}{'(auto)' if t.is_generated else ''}"
                for t in transcript_list
            ) or "none"
            raise TranscriptError(
                f"no transcript for video {video_id} in languages {list(langs)}; "
                f"available: {available}"
            ) from err

    fetched = transcript.fetch()

    return Transcript(
        video_id=fetched.video_id,
        language=fetched.language,
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
        snippets=tuple(
            Snippet(text=s.text, start=s.start, duration=s.duration)
            for s in fetched.snippets
        ),
    )
