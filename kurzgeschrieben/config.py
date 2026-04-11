"""Configuration loader.

`config.yaml` and `.env` are resolved from the current working directory —
the directory the user invoked the CLI (or web app) from. This means you
clone the repo, copy `.env.example` to `.env`, edit it, and run the tool
from that directory. The package itself carries no config; it reads from
wherever you're standing.

A future packaged build can look in `$XDG_CONFIG_HOME/kurzgeschrieben/` first
and fall back to CWD; the loader is intentionally structured so that path
lookup is the only thing that will need to change.

Secrets are NOT read from this file — the API key comes from the environment
(via `.env` + python-dotenv) so the YAML can be committed or shared without
leaking credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _default_config_path() -> Path:
    """Where to look for `config.yaml` by default.

    Resolved lazily (per call) rather than frozen at import time so the file
    is found relative to wherever the user is when they invoke the tool, not
    relative to wherever the package happens to be installed.
    """
    return Path.cwd() / "config.yaml"


def _default_env_path() -> Path:
    return Path.cwd() / ".env"


class ConfigError(Exception):
    """Raised when config is malformed or a required secret is missing."""


@dataclass(frozen=True)
class LLMConfig:
    """LLM / provider settings.

    `model` uses OpenRouter's provider/model slug (e.g. `google/gemini-2.5-flash`,
    `anthropic/claude-sonnet-4.5`). `api_key_env` names the environment variable
    the key is read from — NOT the key itself.
    """

    model: str = "google/gemini-2.5-flash"
    temperature: float = 0.3
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    # None = let the provider pick its default completion length.
    max_tokens: int | None = None


@dataclass(frozen=True)
class SummarizeConfig:
    """Summarization defaults."""

    default_style: str = "wiki_lede"


@dataclass(frozen=True)
class WebConfig:
    """Local web-app settings.

    `allow_regenerate` controls whether the web UI exposes a button (and the
    backend route) for re-running the LLM on an already-summarized video. Set
    this to false before handing the app off to other people so they can't
    spend your credits by spamming regenerate.
    """

    allow_regenerate: bool = True


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    summarize: SummarizeConfig = field(default_factory=SummarizeConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def resolved_api_key(self) -> str:
        """Return the LLM API key from the environment, or raise ConfigError.

        `.env` (in the current working directory) is loaded before lookup, so
        users can drop the key there instead of exporting it in their shell.
        """
        load_dotenv(_default_env_path(), override=False)
        key = os.environ.get(self.llm.api_key_env, "").strip()
        if not key:
            raise ConfigError(
                f"missing API key: set ${self.llm.api_key_env} (e.g. in .env)"
            )
        return key


def load_config(path: Path | None = None) -> Config:
    """Load `config.yaml` and merge it on top of defaults.

    A missing file is not an error — the built-in defaults are used. Unknown
    top-level or nested keys raise `ConfigError` so typos in the YAML become
    loud instead of silently ignored.
    """
    config_path = path or _default_config_path()

    if not config_path.exists():
        return Config()

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigError(
            f"{config_path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )

    llm = _build_llm(raw.pop("llm", {}) or {}, source=config_path)
    summarize = _build_summarize(raw.pop("summarize", {}) or {}, source=config_path)
    web = _build_web(raw.pop("web", {}) or {}, source=config_path)

    if raw:
        raise ConfigError(
            f"{config_path}: unknown top-level keys: {sorted(raw)}"
        )

    return Config(llm=llm, summarize=summarize, web=web)


def _build_llm(data: dict[str, Any], source: Path) -> LLMConfig:
    _reject_unknown(data, LLMConfig, source=source, section="llm")
    return LLMConfig(**data)


def _build_summarize(data: dict[str, Any], source: Path) -> SummarizeConfig:
    _reject_unknown(data, SummarizeConfig, source=source, section="summarize")
    return SummarizeConfig(**data)


def _build_web(data: dict[str, Any], source: Path) -> WebConfig:
    _reject_unknown(data, WebConfig, source=source, section="web")
    return WebConfig(**data)


def _reject_unknown(
    data: dict[str, Any], cls: type, *, source: Path, section: str
) -> None:
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    extra = set(data) - allowed
    if extra:
        raise ConfigError(
            f"{source}: unknown keys under `{section}`: {sorted(extra)} "
            f"(allowed: {sorted(allowed)})"
        )
