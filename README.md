# llm-openai-codex

LLM plugin for accessing ChatGPT/Codex-backed OpenAI models through the Responses API.

This project is forked from and based on Simon Willison's `llm-openai-via-codex`.

## Changes from the original plugin

- Package renamed to `llm-openai-codex`.
- Model prefix changed to `codex/`.
- Plugin-owned auth is stored in LLM's user config directory as `auth-codex.json`, with read-only fallback to `${CODEX_HOME:-~/.codex}/auth.json`.
- `llm codex` auth commands manage login, import, status, refresh, and logout.
- Explicit `verbosity` option maps to Responses API `text.verbosity`.
- Extra Responses API options are forwarded when LLM accepts them.
- Missing `account_id` values are derived from OAuth JWT claims when possible.
- Added `llm codex usage` showing current Codex plan usage.
- Registers known hidden/fallback Codex models such as `gpt-5.3-codex-spark`.

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

Run a prompt:

```bash
llm -m codex/gpt-5.3-codex-spark "Hello"
```

Use Responses API verbosity:

```bash
llm -m codex/gpt-5.3-codex-spark -o verbosity low "Summarize this"
```

## Auth commands

```bash
llm codex login
llm codex login --device-code
llm codex status
llm codex refresh
llm codex usage
llm codex import
llm codex logout
```

Auth source order:

1. Plugin-owned `auth-codex.json`
2. Read-only Codex CLI auth at `${CODEX_HOME:-~/.codex}/auth.json`

`llm codex import` copies ChatGPT OAuth tokens from Codex CLI auth into plugin-owned `auth-codex.json`. It refuses to overwrite an existing `auth-codex.json`; run `llm codex logout` first if you want to replace plugin-owned auth.

When using Codex CLI auth fallback, this plugin does not refresh or delete the shared Codex CLI auth file. If those tokens expire, run Codex CLI to refresh them, or run `llm codex login [--device-code]` to create plugin-owned auth. `llm codex status` shows which auth source is active.

When authentication is missing or expired, run `llm codex login [--device-code]` or `llm codex import`.

## Development

```bash
uv run pytest
uv run llm plugins
uv run llm codex status
```
