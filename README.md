# llm-openai-codex

LLM plugin for accessing ChatGPT/Codex-backed OpenAI models through the Responses API.

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

Install the plugin in the same environment as [LLM](https://llm.datasette.io/):

```bash
llm install llm-openai-codex
```

## Authentication: plugin-owned or borrowed

The plugin can use ChatGPT OAuth credentials from two places, in this order:

1. **Plugin-owned auth** at `auth-codex.json` in LLM's user config directory. This is used whenever the file exists. Create it with `llm codex login`, or copy existing Codex CLI credentials into it with `llm codex import`.
2. **Borrowed Codex CLI auth** at `${CODEX_HOME:-~/.codex}/auth.json`. If plugin-owned auth does not exist, the plugin automatically uses this file; no import is required.

Choose how to set up plugin-owned auth:

```bash
# Perform a new OAuth login and save its credentials for this plugin
llm codex login [--device-code]

# Reuse an existing Codex CLI login by copying its current credentials
llm codex import
```

`login` performs a new OAuth flow. `import` performs no login: it copies the current Codex CLI tokens into `auth-codex.json`, so later plugin refreshes update the plugin's copy instead of the Codex CLI file. Import creates separate storage, not a fresh OAuth authorization. It is useful when you are already logged in with Codex CLI but do not want the plugin to keep writing to its auth file. It refuses to overwrite existing plugin-owned auth; run `llm codex logout` first to replace an imported copy.

Both sources refresh when a token nears expiry and with `llm codex refresh`. Refreshing rotates the refresh token. In borrowed mode this modifies the file shared with Codex CLI and can briefly disrupt a simultaneous CLI session; the plugin prints a warning. `llm codex status` shows the active source. `llm codex logout` deletes plugin-owned auth only, making Codex CLI auth the fallback again.

## Plugin commands

```bash
llm codex login [--device-code]  # create plugin-owned auth with OAuth
llm codex import                 # copy Codex CLI auth into plugin-owned auth
llm codex status                 # show the active source
llm codex refresh                # refresh the active source
llm codex usage                  # show current Codex plan usage
llm codex logout                 # delete plugin-owned auth only
```

## Usage

List available models:

```bash
llm models -q codex
```

Models are discovered from the Codex API and combined with known hidden/fallback slugs. Availability depends on your plan; a listed fallback may still fail at request time. Discovery is cached for 24 hours in `codex_models.json` in LLM's user config directory; delete it to force a re-fetch.

Run a prompt and optionally set Responses API verbosity:

```bash
llm -m codex/gpt-5.6-luna "Hello"
llm -m codex/gpt-5.6-terra -o verbosity low "Summarize this"
```

Image attachments (`-a image.png`) are sent at low detail to limit token use.

## Web search

Enable OpenAI's server-side search (it runs on OpenAI's servers, not locally):

```bash
llm -m codex/gpt-5.6-luna -o web_search 1 "What happened in AI news today?"
llm -m codex/gpt-5.6-luna -o web_search 1 -o web_search_live 1 "Latest stable Python release?"
llm -m codex/gpt-5.6-luna -o web_search 1 -o web_search_context_size high "Recent LLM benchmarks"
```

`web_search_live` requests live internet access instead of the cached index. Context size is `low`, `medium`, or `high`. Availability depends on the plan and model. Set a per-model default with:

```bash
llm models options set codex/gpt-5.6-luna web_search 1
```

## Development

```bash
uv run pytest
uv run llm plugins
uv run llm codex status
```

## Releasing

1. Finish and verify the feature/fix commits.
2. Bump `version` in `pyproject.toml` (without `v`), then commit it separately: `git add pyproject.toml && git commit -m "bump: v0.4.2"`.
3. Tag and push that commit: `git tag v0.4.2 && git push origin main v0.4.2`.
4. Run `gh release create v0.4.2 --generate-notes`; publication triggers testing and PyPI publication.
