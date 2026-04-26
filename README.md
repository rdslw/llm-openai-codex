# llm-openai-codex

LLM plugin for accessing ChatGPT/Codex-backed OpenAI models through the Responses API.

This project is forked from and based on Simon Willison's `llm-openai-via-codex`.

## Changes from the original plugin

- Package renamed to `llm-openai-codex`.
- Model prefix changed to `codex/`.
- Plugin-owned auth is stored in LLM's user config directory as `auth-codex.json`.
- `llm codex` auth commands manage login, import, status, refresh, and logout.
- Explicit `verbosity` option maps to Responses API `text.verbosity`.
- Extra Responses API options are forwarded when LLM accepts them.
- Missing `account_id` values are derived from OAuth JWT claims when possible.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-openai-codex
```

## Usage

Authenticate the plugin:

```bash
llm codex login
```

List available Codex-backed models:

```bash
llm models -q codex
```

Run a prompt:

```bash
llm -m codex/gpt-5.5 "Hello"
```

Use Responses API verbosity:

```bash
llm -m codex/gpt-5.5 -o verbosity low "Summarize this"
```

## Auth commands

```bash
llm codex login
llm codex login --device-code
llm codex status
llm codex refresh
llm codex import
llm codex logout
```

`llm codex import` copies ChatGPT OAuth tokens from `${CODEX_HOME:-~/.codex}/auth.json` into the plugin-owned `auth-codex.json`. Normal model calls read only the plugin-owned auth file.

## Development

```bash
uv run pytest
uv run llm plugins
uv run llm codex status
```
