"""kurzgeschrieben — turn YouTube videos into something you can read.

Usage:

    kurzgeschrieben <url-or-video-id>              # fetch + summarize
    kurzgeschrieben <url> --raw                    # just the transcript
    kurzgeschrieben <url> --raw --json             # transcript as JSON
    kurzgeschrieben <url> --refresh                # bypass the cache
    kurzgeschrieben <url> --style wiki_lede        # pick a summary style
    kurzgeschrieben <url> --lang en,de             # language priority

Transcripts are cached locally so re-running against a different summary
style (or just rereading) costs nothing and skips the network entirely.
"""

from __future__ import annotations

import argparse
import json
import sys

from .cache import get_transcript
from .config import ConfigError, load_config
from .llm import LLMError, available_styles, requires_video_meta, summarize
from .transcript import TranscriptError
from .youtube_meta import fetch_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kurzgeschrieben",
        description="Download YouTube transcripts and summarize them for reading.",
    )
    parser.add_argument(
        "url",
        help="YouTube URL or bare 11-char video id",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Comma-separated language priority list (default: en)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a fresh fetch even if the transcript is cached",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip the LLM and print the transcript itself",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --raw: emit JSON (metadata + timestamped snippets)",
    )
    parser.add_argument(
        "--style",
        default=None,
        choices=available_styles(),
        help="Summary style (default: from config.yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.json and not args.raw:
        print("error: --json only makes sense with --raw", file=sys.stderr)
        return 2

    languages = tuple(code.strip() for code in args.lang.split(",") if code.strip())

    try:
        config = load_config()
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return 1

    try:
        transcript, from_cache = get_transcript(
            args.url,
            languages=languages,
            refresh=args.refresh,
        )
    except TranscriptError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    source = "auto-generated" if transcript.is_generated else "manual"
    cache_tag = "cache" if from_cache else "fresh"
    header = (
        f"# {transcript.video_id}  "
        f"({transcript.language_code}, {source}, {cache_tag})"
    )

    if args.raw:
        if args.json:
            payload = {
                "video_id": transcript.video_id,
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "snippets": [
                    {"text": s.text, "start": s.start, "duration": s.duration}
                    for s in transcript.snippets
                ],
            }
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            print(header, file=sys.stderr)
            print(transcript.plain_text)
        return 0

    # Default path: fetch (cache-first) + summarize via LLM.
    chosen_style = args.style or config.summarize.default_style
    # wiki_article (and any future style that embeds video metadata in the
    # prompt) needs one extra network call to oEmbed for the title/creator.
    meta = fetch_meta(transcript.video_id) if requires_video_meta(chosen_style) else None

    try:
        summary = summarize(transcript, style=chosen_style, config=config, meta=meta)
    except (LLMError, ConfigError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(header, file=sys.stderr)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
