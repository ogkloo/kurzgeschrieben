"""LLM-backed transcript summarization.

The goal is reading-first consumption of video content. The default output
style is `wiki_lede` — a Wikipedia-article-opening-section summary of whatever
the video is actually *about*, not a meta-summary of the video itself.

OpenRouter is used as the LLM gateway because it exposes a large model catalog
behind one OpenAI-compatible endpoint, which lets users swap models via
config.yaml without any code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from openai import OpenAI, OpenAIError

from .config import Config, load_config
from .transcript import Transcript
from .youtube_meta import VideoMeta


class LLMError(Exception):
    """Raised when an LLM request fails or is misconfigured."""


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user_template: str  # formatted with transcript_text=


# A Wikipedia "lede" is the opening section of an article: a few short
# paragraphs that define the subject, give the most important facts, and let a
# reader who reads *only* the lede come away with a complete-enough basic
# understanding. That is exactly the shape we want for infotainment videos —
# it's compact, factual, reorganized topically (not by runtime), and neutral.
WIKI_LEDE_SYSTEM: Final[str] = """\
You write Wikipedia-style lede sections (the opening paragraphs of a Wikipedia article) from a source transcript.

A Wikipedia lede:
- Opens with a single sentence that defines the subject.
- Covers the most important facts first: what it is, who is involved, when and where it happened (if relevant), and why it matters.
- Is written in neutral, factual encyclopedic prose.
- Is standalone: a reader who reads only the lede should come away with a complete basic understanding of the subject.
- Is short — typically 1 to 3 paragraphs, roughly 150 to 350 words.

Important rules for this task:
1. The subject of the lede is whatever the video is ABOUT, not the video itself. If the video is a review of X, the subject is X. If the video explains a concept Y, the subject is Y. Never write "This video..." or "The speaker says..." — write directly about the subject.
2. Use only facts present in the transcript. Do not invent dates, names, numbers, or context that isn't there. If the transcript is too shallow, anecdotal, or rambling to support a real lede, respond with a single sentence explaining that instead of fabricating one.
3. Reorganize facts by importance and topic, not by when they appeared in the video. The video's narrative order is usually wrong for a lede.
4. Output ONLY the lede prose. No headings, no bullet points, no "Summary:" prefix, no meta-commentary about the video or the transcript."""

WIKI_LEDE_USER_TEMPLATE: Final[str] = """\
Here is the transcript of a YouTube video. Write a Wikipedia-style lede about the subject of the video.

--- TRANSCRIPT ---
{transcript_text}
--- END TRANSCRIPT ---"""


# Full Wikipedia-style article: a lede paragraph followed by topical body
# sections under `## Heading` marks. The faithfulness rules here are
# significantly stricter than for wiki_lede because a longer output gives the
# model a much larger surface on which to pad with training-data knowledge.
# The rules explicitly give the model an escape hatch (omit sections, bail out
# entirely) so it doesn't feel pressured to invent detail to fill space.
#
# This prompt also asks the model to:
#   (A1) write a short `## About this video` first body section that addresses
#        the video as a *source* — creator, length, genre, thesis — so the
#        reader comes away knowing what the video actually is;
#   (B1) derive the remaining body sections from the video's own pacing by
#        reading the `[M:SS]` timestamps on each transcript line, so the
#        article's structure mirrors the video's structure rather than some
#        abstract taxonomy the LLM invented.
WIKI_ARTICLE_SYSTEM: Final[str] = """\
You write full Wikipedia-style articles from a timestamped video transcript.

INPUT FORMAT:
You receive a video-metadata block (title, creator, length) and a transcript where every line is prefixed with `[M:SS]` indicating when that line begins in the video. Use those timestamps to sense the video's natural segmentation — they show where the creator transitions between topics.

ARTICLE STRUCTURE:

- Open with a lede of 1 to 3 short paragraphs. The lede has NO heading — it is just prose at the top of the document. It defines the subject and summarizes the most important facts so a reader who reads only the lede has a complete basic understanding.

- The FIRST body section must be titled exactly `## About this video`. It is short — 2 to 3 sentences. Name the creator (from the metadata block), state the approximate length rounded naturally (e.g. "about ten minutes"), identify the kind of video it is (explainer, breakdown, analysis, review, commentary, deep dive), and describe the creator's angle, thesis, or approach — what the video is actually arguing or demonstrating. This section is the ONLY place in the article where you write about the video itself.

- The REMAINING body sections come from the video's own structure. Read the `[M:SS]` timestamps to identify where the video transitions between distinct topics or phases (setup, background, the main claim, examples, tradeoffs, wrap-up, etc.) and let your section headings correspond to those transitions. Section titles should describe what the video covers in each segment, not impose an abstract outside taxonomy. Keep the sections in roughly the video's order unless a drastically different order is clearer for a reader encountering the material cold.

- Use `### Subsection` headings inside a section only when the material genuinely needs further breakdown.

- Write neutral, factual encyclopedic prose. Avoid bullet lists except for genuinely enumerable items (a list of named works, a list of species). Do not use bullets to avoid writing sentences.

- Reorder individual facts within a section by importance when it helps the reader. The finished article should stand alone as a basic reference on the subject.

RULES ABOUT WHAT YOU MAY WRITE. These override everything else:

1. Voice and scope. The "About this video" section is about the video itself: you may name the creator, describe their approach, and write sentences like "The video argues that…" or "The creator presents…". EVERY OTHER SECTION — including the lede — is about the subject matter. In those sections, never write "This video…", "The speaker argues…", "In this talk…". Write directly about the subject as if composing an encyclopedia entry.

2. You may use ONLY facts present in the transcript or the metadata block. Do not add dates, names, nationalities, numerical figures, institutional affiliations, technical terms, or historical context that are not in the source, even if you are confident they are true. This is the most important rule in the entire prompt.

3. Before finalizing each sentence, check it against the source. If a claim in your draft is not traceable to something in the transcript or metadata, remove it or rewrite it until it is. Shorter and accurate beats longer and padded.

4. If the transcript does not contain enough material to support a body section you were planning, omit that section entirely. Do not pad a weak section; delete it.

5. If the transcript is fundamentally too shallow to sustain any real article — for example it is song lyrics, casual chat, or a single anecdote — respond with one short paragraph explaining that instead of fabricating an article. This is the correct answer for thin source material.

6. Do NOT include the `[M:SS]` timestamps in your output. They are only for your own reading of the source. The finished article is plain Wikipedia-style prose and markdown headings.

7. Output ONLY the finished article in markdown. No preamble like "Here is the article". No trailing commentary. No "Summary:" prefix. Start with the first sentence of the lede."""

WIKI_ARTICLE_USER_TEMPLATE: Final[str] = """\
Here is the video metadata and the timestamped transcript. Write a Wikipedia-style article about the subject of the video, following the structure and rules in the system message.

--- VIDEO METADATA ---
Title: {title}
Creator: {author}
Length: {duration_display}
--- END METADATA ---

--- TRANSCRIPT (timestamped) ---
{transcript_text}
--- END TRANSCRIPT ---"""


INTERVIEW_SYSTEM: Final[str] = """\
You write interview summaries from timestamped video transcripts.

Your job is to extract and organize the key information from an interview or conversation, focusing on who the participants are, what each person contributes, and the most important insights.

ARTICLE STRUCTURE:

- Open with 1 to 2 paragraphs identifying the interview: who is talking, what the conversation is about, and why it matters. This section has NO heading — it is just prose at the top of the document.

- Next, a section titled exactly `## Participants`. For each speaker you can identify from the transcript, write a short paragraph covering their name, role, expertise, or perspective as far as the transcript reveals. If the transcript does not clearly identify speakers, describe them by their apparent role or viewpoint (e.g. "the host", "the guest", "the researcher"). Do not invent credentials or biographical details that are not in the source.

- The remaining sections organize the interview's substance by topic, not by chronological order. Each `## Heading` should name the topic or theme discussed. Within each section:
  - Attribute key claims, arguments, and observations to specific speakers using phrases like "According to [name]…" or "[Name] argues that…".
  - Highlight points of genuine insight — things that are surprising, counter-intuitive, or that reflect deep expertise.
  - Note significant disagreements or contrasting perspectives between speakers when they occur.

- Use `### Subsection` headings only when a topic genuinely needs further breakdown.

RULES — these override everything else:

1. Attribution is the priority. Every substantial claim should be tied to the person who made it. Generic summaries that lose track of who said what defeat the purpose of this format.

2. Use ONLY facts present in the transcript or the metadata block. Do not add dates, names, nationalities, credentials, institutional affiliations, or context that are not in the source, even if you are confident they are true.

3. Focus on insight density. Surface the most interesting, actionable, or surprising points. Skip small talk, repetitive points, and filler. Shorter and accurate beats longer and padded.

4. Write in clear, factual prose — closer to a briefing document than a narrative retelling. Avoid bullet lists except for genuinely enumerable items.

5. If the transcript is not actually an interview or conversation (e.g. a solo explainer, a music video, a monologue), say so in one short paragraph and then write a standard topical summary of whatever the content actually is. Do not force the interview format onto non-interview material.

6. Do NOT include `[M:SS]` timestamps in your output. They are only for your own reading of the source.

7. Output ONLY the finished summary in markdown. No preamble like "Here is the summary". No trailing commentary. Start with the first sentence of the opening paragraphs."""

INTERVIEW_USER_TEMPLATE: Final[str] = """\
Here is the video metadata and the timestamped transcript. Write an interview summary following the structure and rules in the system message.

--- VIDEO METADATA ---
Title: {title}
Creator: {author}
Length: {duration_display}
--- END METADATA ---

--- TRANSCRIPT (timestamped) ---
{transcript_text}
--- END TRANSCRIPT ---"""


PROMPTS: Final[dict[str, PromptTemplate]] = {
    "wiki_lede": PromptTemplate(
        system=WIKI_LEDE_SYSTEM,
        user_template=WIKI_LEDE_USER_TEMPLATE,
    ),
    "wiki_article": PromptTemplate(
        system=WIKI_ARTICLE_SYSTEM,
        user_template=WIKI_ARTICLE_USER_TEMPLATE,
    ),
    "interview": PromptTemplate(
        system=INTERVIEW_SYSTEM,
        user_template=INTERVIEW_USER_TEMPLATE,
    ),
}


def available_styles() -> list[str]:
    return sorted(PROMPTS.keys())


# Styles that expect a `VideoMeta` + timestamped transcript to be piped into
# the user prompt. Anything not listed here takes the simpler
# `{transcript_text}`-only shape.
_META_REQUIRING_STYLES: Final[set[str]] = {"wiki_article", "interview"}


def requires_video_meta(style: str) -> bool:
    """Whether callers need to fetch oEmbed metadata before calling summarize().

    Exposed so the CLI and web app can decide whether to pay the extra network
    round-trip without having to duplicate the membership check here.
    """
    return style in _META_REQUIRING_STYLES


def _format_duration(seconds: float) -> str:
    """Render a duration in seconds as `M:SS` or `H:MM:SS`.

    This is what gets dropped into the metadata block for wiki_article so the
    model can phrase the length naturally ("about ten minutes", "just over an
    hour") without needing to do arithmetic itself.
    """
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def summarize(
    transcript: Transcript,
    *,
    style: str | None = None,
    config: Config | None = None,
    meta: VideoMeta | None = None,
) -> str:
    """Summarize a transcript in the requested style via the configured LLM.

    `style` defaults to the value in `config.summarize.default_style`.
    `config` defaults to the result of `load_config()`.
    `meta` is required for styles that include video metadata in the prompt
    (currently `wiki_article`); passing it for a style that doesn't need it is
    fine — it's simply ignored.

    Returns the model's text response.
    """
    if config is None:
        config = load_config()

    chosen_style = style or config.summarize.default_style
    if chosen_style not in PROMPTS:
        raise LLMError(
            f"unknown style {chosen_style!r}; available: {available_styles()}"
        )

    if chosen_style in _META_REQUIRING_STYLES and meta is None:
        raise LLMError(
            f"style {chosen_style!r} requires video metadata; "
            f"caller must pass `meta=VideoMeta(...)`"
        )

    if chosen_style in _META_REQUIRING_STYLES:
        source_text = transcript.timestamped_text().strip()
    else:
        source_text = transcript.plain_text.strip()

    if not source_text:
        raise LLMError(
            f"transcript for video {transcript.video_id} has no text to summarize"
        )

    prompt = PROMPTS[chosen_style]
    format_vars: dict[str, str] = {"transcript_text": source_text}
    if chosen_style in _META_REQUIRING_STYLES:
        assert meta is not None  # guaranteed by the check above
        format_vars.update(
            title=meta.title,
            author=meta.author or "unknown",
            duration_display=_format_duration(transcript.duration_seconds),
        )
    user_message = prompt.user_template.format(**format_vars)

    api_key = config.resolved_api_key()  # raises ConfigError if unset
    client = OpenAI(base_url=config.llm.base_url, api_key=api_key)

    create_kwargs: dict = {
        "model": config.llm.model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": user_message},
        ],
        "temperature": config.llm.temperature,
    }
    if config.llm.max_tokens is not None:
        create_kwargs["max_tokens"] = config.llm.max_tokens

    try:
        response = client.chat.completions.create(**create_kwargs)
    except OpenAIError as err:
        raise LLMError(f"LLM request failed: {err}") from err

    if not response.choices:
        raise LLMError("LLM response contained no choices")

    content = response.choices[0].message.content
    if not content:
        raise LLMError("LLM response was empty")

    return content.strip()
