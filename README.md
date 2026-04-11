# kurzgeschrieben

Turn YouTube videos into Wikipedia-style articles. Downloads the transcript,
feeds it to a configurable LLM, and writes a `## About this video` section
plus a full article whose structure follows the video's own pacing. Ships a
CLI and a local web app with a cleaner-Wikipedia dark-mode reader.

Built for people who read much faster than they watch video.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for dependency management and running the scripts)
- An [OpenRouter](https://openrouter.ai) API key — the default model is
  `google/gemini-2.5-flash`, which is cheap enough that dogfooding a few
  dozen summaries costs pennies. Any OpenAI-compatible endpoint works; see
  the config section below.

## Install

```sh
git clone https://github.com/ogkloo/kurzgeschrieben.git
cd kurzgeschrieben
uv sync
cp .env.example .env
$EDITOR .env   # paste your OPENROUTER_API_KEY
```

That's it. `uv sync` installs the package in editable mode, which registers
the `kurzgeschrieben` and `kurzgeschrieben-web` commands inside the project's
virtualenv.

## CLI usage

```sh
# Summarize a video (uses the default style from config.yaml)
uv run kurzgeschrieben 'https://www.youtube.com/watch?v=...'

# Full Wikipedia-style article instead of just the lede
uv run kurzgeschrieben --style wiki_article 'https://youtu.be/...'

# Dump the raw transcript (no LLM call)
uv run kurzgeschrieben --raw 'https://www.youtube.com/watch?v=...'

# Raw transcript as JSON with timestamps
uv run kurzgeschrieben --raw --json 'https://www.youtube.com/watch?v=...'

# Force a fresh fetch, bypassing the on-disk cache
uv run kurzgeschrieben --refresh 'https://www.youtube.com/watch?v=...'

# Non-English priority (first match wins)
uv run kurzgeschrieben --lang de,en 'https://www.youtube.com/watch?v=...'
```

URLs can be `youtube.com/watch?v=...`, `youtu.be/...`, `shorts/...`,
`embed/...`, `live/...`, or a bare 11-character video id.

## Web app

```sh
uv run kurzgeschrieben-web
```

Opens on <http://127.0.0.1:7777>. Paste a URL, wait ~10-20 seconds, get a
dark-mode Wikipedia-style page with a lede, table of contents, and body
sections. The index page lists every summary you've generated so far.
Re-submitting an already-summarized URL is instant (cached).

The web app uses the `wiki_article` style regardless of the CLI default.

### Sharing the web app with other people

The `Regenerate` button spends API credits on every click, so if you hand the
app to someone else you almost certainly want it off. Set:

```yaml
# config.yaml
web:
  allow_regenerate: false
```

This hides the button AND makes the backend route return 403, so a curious
user can't call it directly either.

Note: the server binds to `127.0.0.1` by default — it's only reachable from
the machine running it. To expose it to your LAN, edit `kurzgeschrieben-web`'s
`run(host=...)` or wrap it in a small script. Be mindful that there is no
auth layer; don't put this on the open internet as-is.

## Configuration

`config.yaml` lives at the root of whichever directory you invoke the tool
from (usually the repo clone). Missing file = sensible defaults. Unknown keys
are loud errors, not silently ignored.

```yaml
llm:
  model: google/gemini-2.5-flash
  temperature: 0.3
  # max_tokens: 4096           # uncomment to cap completion length
  # base_url: https://...      # for non-OpenRouter OpenAI-compatible gateways
  # api_key_env: OPENROUTER_API_KEY  # env var name to read the key from

summarize:
  # Available styles:
  #   wiki_lede     — opening section only (1-3 short paragraphs)
  #   wiki_article  — full article with lede + `## Section` body headings
  default_style: wiki_lede

web:
  allow_regenerate: true
```

### Model selection

The `llm.model` slug uses OpenRouter's `provider/model` format. A few useful
swaps:

| slug                            | notes                                    |
|---------------------------------|------------------------------------------|
| `google/gemini-2.5-flash`       | cheap and fast; current default          |
| `google/gemini-2.5-pro`         | stronger, still inexpensive              |
| `anthropic/claude-sonnet-4.5`   | strongest, more expensive                |
| `openai/gpt-5`                  | similar tier to Sonnet                   |

Browse the full catalog at <https://openrouter.ai/models>.

### Non-OpenRouter providers

Any OpenAI-compatible endpoint works — point `base_url` at the gateway and
`api_key_env` at the env var holding the key. A local Ollama server, for
example:

```yaml
llm:
  model: llama3.1:8b
  base_url: http://localhost:11434/v1
  api_key_env: OLLAMA_API_KEY  # put any non-empty value in this env var
```

## Where data lives

All paths are relative to the directory you invoke the tool from:

- `.env` — `OPENROUTER_API_KEY` (gitignored, never commit)
- `config.yaml` — user settings (safe to commit; no secrets in it)
- `.cache/transcripts/{video_id}.json` — fetched transcripts with timestamps
- `.cache/summaries/{video_id}__{style}.json` — generated articles

The caches are the primary source of truth: re-running the CLI or web app
against an already-seen video never hits YouTube, and re-using an already-
generated summary never hits the LLM. Both caches are schema-versioned —
stale files from older releases are treated as misses, never crashes.

## Output quality and hallucinations

The `wiki_article` prompt leans hard on faithfulness: don't invent dates,
names, figures, or context that isn't in the transcript. In practice this
holds well for short and moderately-technical videos and leaks on long,
well-documented topics where the model has a lot to pull from training data.
Concretely, on a ~10-minute explainer the output is essentially faithful; on
a 46-minute technical talk about CPU caches it starts slotting in plausible-
but-unsourced latency numbers.

If you care about correctness more than the current prompt delivers, the
next lever is a second-pass check (feed the draft + transcript back to the
same model and ask it to strike any sentence not traceable to the source).
Not implemented yet.

## Limitations

- Single-user, single-machine. No auth, no job queue, no multi-tenancy.
- English-default. Non-English transcripts work via `--lang`, but the prompt
  itself is written in English and asks for English output.
- Auto-generated captions work but are noisier than human-written ones; the
  transcript fetcher always prefers manual captions when both exist.
- Transcripts disabled on the video = nothing to do. This tool has no speech-
  to-text fallback.

## License

MIT — see [LICENSE](LICENSE).
