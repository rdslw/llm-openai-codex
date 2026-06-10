# llm-openai-codex

LLM plugin for accessing ChatGPT/Codex-backed OpenAI models through the Responses API.

Fork of Simon Willison's `llm-openai-via-codex` with a `codex/` model prefix, plugin-owned auth, `llm codex` commands, verbosity, and OpenAI's server-side `web_search` tool.

## Changes from the original plugin

- Package renamed to `llm-openai-codex`.
- Model prefix changed to `codex/`.
- Plugin-owned auth is stored in LLM's user config directory as `auth-codex.json`, with borrowed fallback to `${CODEX_HOME:-~/.codex}/auth.json`.
- `llm codex` auth commands manage login, import, status, refresh, and logout.
- Explicit `verbosity` option maps to Responses API `text.verbosity`.
- Extra Responses API options are forwarded when LLM accepts them.
- Missing `account_id` values are derived from OAuth JWT claims when possible.
- Added `llm codex usage` showing current Codex plan usage.
- Registers known hidden/fallback Codex models such as `gpt-5.3-codex-spark`.
- Added OpenAI's server-side `web_search` tool via `-o web_search 1`.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-openai-codex
```

## Usage

Authenticate the plugin:

```bash
llm codex login [--device-code]
```

If you already use Codex CLI, the plugin can fall back to `${CODEX_HOME:-~/.codex}/auth.json` when `auth-codex.json` does not exist. You can also copy those tokens into plugin-owned auth with `llm codex import`.

List available Codex-backed models:

```bash
llm models -q codex
```

Models are discovered from the Codex API, plus known hidden/fallback slugs such as `gpt-5.3-codex-spark`. Some listed fallback models may still fail at request time if your plan lacks access.

The discovered list is cached for 24 hours in `codex_models.json` in LLM's user config directory; delete that file to force a re-fetch.

Run a prompt:

```bash
llm -m codex/gpt-5.3-codex-spark "Hello"
```

Use Responses API verbosity:

```bash
llm -m codex/gpt-5.3-codex-spark -o verbosity low "Summarize this"
```

## Web search

Enable OpenAI's server-side `web_search` tool (the search runs on OpenAI's servers, not locally):

```bash
llm -m codex/gpt-5.5 -o web_search 1 "What happened in AI news today?"
llm -m codex/gpt-5.5 -o web_search 1 -o web_search_live 1 "Latest stable Python release?"
llm -m codex/gpt-5.5 -o web_search 1 -o web_search_context_size high "Summarize recent LLM benchmarks"
```

`web_search_live` requests live internet access instead of the cached index; `web_search_context_size` is one of `low`, `medium`, `high`. Availability depends on your plan and model.

To enable web search by default for a given model, use LLM's per-model default options:

```bash
llm models options set codex/gpt-5.5 web_search 1
```

## Auth commands

```bash
llm codex login
llm codex login --device-code
llm codex status
llm codex refresh
llm codex refresh --borrowed
llm codex usage
llm codex import
llm codex logout
```

Auth source order:

1. Plugin-owned `auth-codex.json`
2. Borrowed Codex CLI auth at `${CODEX_HOME:-~/.codex}/auth.json`

`llm codex import` copies ChatGPT OAuth tokens from Codex CLI auth into plugin-owned `auth-codex.json`. It refuses to overwrite an existing `auth-codex.json`; run `llm codex logout` first if you want to replace plugin-owned auth.

### Refresh behavior

- **Plugin-owned auth** refreshes lazily and on demand via `llm codex refresh`.
- **Borrowed Codex CLI auth** never refreshes automatically. When it expires, pick one:
  - `llm codex refresh --borrowed` — refresh the shared file in place. **Rotates the shared `refresh_token`; restart any running Codex CLI session afterwards.**
  - Run Codex CLI itself to refresh.
  - `llm codex import` to promote it to plugin-owned auth.

`llm codex status` shows which auth source is active.

When authentication is missing or expired, run `llm codex login [--device-code]` or `llm codex import`.

## Development

```bash
uv run pytest
uv run llm plugins
uv run llm codex status
```

## Releasing

1. Bump `version` in `pyproject.toml` (PEP 440, no `v` prefix, e.g. `0.2.4`).
2. Commit, then tag with a `v` prefix and push:
   ```bash
   git tag v0.2.4 && git push origin v0.2.4
   ```
3. Publish a GitHub Release for that tag (`gh release create v0.2.4 --generate-notes`). This triggers `.github/workflows/publish.yml` to test and publish to PyPI.
