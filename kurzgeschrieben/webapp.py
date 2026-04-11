"""Local web app for kurzgeschrieben.

Routes:

    GET  /                          — submission form + list of cached summaries
    POST /submit                    — run the fetch + summarize pipeline, save, redirect
    GET  /s/<video_id>              — render a cached summary as a cleaner-Wikipedia page
    POST /s/<video_id>/regenerate   — drop the summary cache entry, re-run the LLM pass
                                      (gated by config.web.allow_regenerate — 403 when off)

Request handling is intentionally blocking: Gemini Flash calls typically
return in well under 30 seconds, this server is single-user and local, and
skipping the background-job machinery means there's nothing to get wrong.
Transcript and summary caching make re-submits and reloads instant.

Start with:

    kurzgeschrieben-web

Server listens on http://127.0.0.1:7777 by default.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import markdown
from flask import Flask, abort, redirect, render_template, request, url_for

from . import summary_cache
from . import youtube_meta
from .cache import get_transcript
from .config import Config, ConfigError, load_config
from .llm import LLMError, summarize
from .summary_cache import Summary
from .transcript import TranscriptError, extract_video_id


# Templates and static files ship inside the installed package, so they are
# resolved relative to THIS file, not to the user's working directory. User
# data (config.yaml, .env, caches) is still resolved from cwd by the modules
# that own those paths.
PACKAGE_DIR = Path(__file__).resolve().parent

# The web UI is built around the wiki_article style specifically — that's
# what the user is dogfooding. The CLI still respects the config default.
WEB_STYLE: Final[str] = "wiki_article"

# The ToC below the lede includes h2 and h3. Deeper is noise; shallower hides
# the sub-topics that make navigation worthwhile.
TOC_DEPTH: Final[str] = "2-3"


app = Flask(
    __name__,
    template_folder=str(PACKAGE_DIR / "templates"),
    static_folder=str(PACKAGE_DIR / "static"),
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    summaries = summary_cache.list_all()
    return render_template("index.html", summaries=summaries, error=None)


@app.route("/submit", methods=["POST"])
def submit():
    url = (request.form.get("url") or "").strip()
    if not url:
        return _render_index_error("please paste a YouTube URL")

    try:
        video_id = extract_video_id(url)
    except TranscriptError as err:
        return _render_index_error(str(err))

    # Fast path: already have a cached summary for this (video, style) pair.
    # Re-submitting is meant to be an instant bookmark-jump, not a regenerate.
    if summary_cache.load(video_id, WEB_STYLE) is not None:
        return redirect(url_for("show_summary", video_id=video_id))

    try:
        config = load_config()
    except ConfigError as err:
        return _render_index_error(f"config error: {err}")

    try:
        _run_summary_pipeline(video_id, config)
    except (TranscriptError, LLMError, ConfigError) as err:
        return _render_index_error(str(err))

    return redirect(url_for("show_summary", video_id=video_id))


@app.route("/s/<video_id>")
def show_summary(video_id: str):
    summary = summary_cache.load(video_id, WEB_STYLE)
    if summary is None:
        abort(404)

    try:
        config = load_config()
    except ConfigError as err:
        return _render_index_error(f"config error: {err}")

    lede_html, toc_html, body_html = _render_article_markdown(summary.content)

    return render_template(
        "summary.html",
        summary=summary,
        lede_html=lede_html,
        toc_html=toc_html,
        body_html=body_html,
        allow_regenerate=config.web.allow_regenerate,
    )


@app.route("/s/<video_id>/regenerate", methods=["POST"])
def regenerate_summary(video_id: str):
    try:
        config = load_config()
    except ConfigError as err:
        return _render_index_error(f"config error: {err}")

    if not config.web.allow_regenerate:
        # Route is disabled by config — refuse even if someone knows the URL.
        # This is the server-side enforcement that makes the "hand this to
        # other people" toggle actually safe.
        abort(403)

    # Must currently exist; regenerating a video we've never summarized is a
    # nonsense request (use /submit for that).
    if summary_cache.load(video_id, WEB_STYLE) is None:
        abort(404)

    # Drop the old summary. Transcript cache is untouched — regeneration is
    # purposely an LLM-only re-run, not a YouTube re-fetch.
    summary_cache.delete(video_id, WEB_STYLE)

    try:
        _run_summary_pipeline(video_id, config)
    except (TranscriptError, LLMError, ConfigError) as err:
        return _render_index_error(str(err))

    return redirect(url_for("show_summary", video_id=video_id))


@app.errorhandler(404)
def not_found(_err):
    return (
        render_template(
            "index.html",
            summaries=summary_cache.list_all(),
            error="not found",
        ),
        404,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run_summary_pipeline(video_id: str, config: Config) -> Summary:
    """Fetch transcript (cache-first), fetch metadata, run LLM, save summary.

    Shared by /submit and /regenerate. Metadata must be fetched BEFORE the LLM
    call because `wiki_article` embeds title/creator/length directly into the
    user prompt. A failed oEmbed still returns a best-effort `VideoMeta` (with
    the bare video id as the fallback title) so the pipeline keeps running.

    Raises TranscriptError / LLMError / ConfigError on failure so callers can
    surface a clean error to the user.
    """
    transcript, _ = get_transcript(video_id)
    meta = youtube_meta.fetch_meta(video_id)
    content = summarize(transcript, style=WEB_STYLE, config=config, meta=meta)

    summary = Summary(
        video_id=video_id,
        video_title=meta.title,
        video_author=meta.author,
        style=WEB_STYLE,
        model=config.llm.model,
        content=content,
        created_at=summary_cache.now_iso(),
    )
    summary_cache.save(summary)
    return summary


def _render_article_markdown(content: str) -> tuple[str, str, str]:
    """Split a wiki_article's markdown at the first `## ` heading and render
    each half into HTML. Returns `(lede_html, toc_html, body_html)`.

    - `lede_html`: the prose before the first body heading.
    - `toc_html`: the ToC built from the body's h2/h3 headings, or `""` if
      there are none (which happens when the LLM bailed out with a "too
      shallow" one-paragraph response — no ToC to show).
    - `body_html`: the body sections themselves, with anchor-linked headings
      so ToC links resolve.

    Two separate Markdown instances are used so the toc extension only runs
    on the body. That avoids the toc extension accidentally indexing stray
    headings in the lede and keeps the per-instance state simple.
    """
    split_match = re.search(r"^## ", content, re.MULTILINE)

    if split_match is None:
        # No body headings — all content is lede. Still needs to go through
        # markdown for paragraph/emphasis rendering.
        lede_only = markdown.Markdown().convert(content)
        return lede_only, "", ""

    lede_md = content[: split_match.start()].rstrip()
    body_md = content[split_match.start() :]

    lede_html = markdown.Markdown().convert(lede_md)

    body_md_renderer = markdown.Markdown(
        extensions=["toc"],
        extension_configs={"toc": {"toc_depth": TOC_DEPTH}},
    )
    body_html = body_md_renderer.convert(body_md)
    toc_html = body_md_renderer.toc

    return lede_html, toc_html, body_html


def _render_index_error(message: str) -> str:
    return render_template(
        "index.html",
        summaries=summary_cache.list_all(),
        error=message,
    )


def run(host: str = "127.0.0.1", port: int = 7777) -> None:
    """Entry point for the `kurzgeschrieben-web` console script.

    Binds to loopback by default so the server is reachable only from the
    machine it's running on — this is meant as a local-only dogfooding tool,
    not a public service. Pass `host="0.0.0.0"` to expose it to your LAN.
    """
    app.run(host=host, port=port)


if __name__ == "__main__":
    run()
