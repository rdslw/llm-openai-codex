# Repository guidance

## Scope and layout

- `llm_openai_codex.py` contains the plugin implementation, model registration,
  Responses API integration, authentication, model discovery, usage reporting,
  and `llm codex` commands.
- `tests/test_openai_codex.py` is the main test suite.
- `README.md` is the user-facing source of truth for installation and behavior.

## Development

- Use `uv` for the development environment.
- Run the full suite with `uv run pytest` after code changes. Do not run
  `.venv/bin/python -m pytest` instead: it skips the sync of the environment
  with `pyproject.toml`.
- In a sandbox, the default uv cache (`~/.cache/uv`) is not writable. Prefix
  each `uv` command with a cache in the sandbox temporary directory:
  `UV_CACHE_DIR="$TMPDIR/uv-cache" UV_LINK_MODE=copy uv run pytest`.
  `UV_LINK_MODE=copy` stops the hardlink warning (the cache and `.venv` are on
  different file systems). The first run downloads the packages again.
- Add or update tests for behavior changes, including error and fallback paths.
- Keep synchronous and asynchronous model behavior aligned through the shared
  implementation.

## Model invariants

- Public LLM model IDs use the `codex/` prefix.
- Request input is built from `prompt.messages`, the canonical full chain
  since llm 0.32. Never walk `conversation.responses`; reloaded conversations
  no longer populate it.
- Reasoning metadata (`id`, `encrypted_content`, `summary`) round-trips via
  `provider_metadata["openai"]` on `ReasoningPart` objects; requests stay
  stateless with `store: false`.
- A request sends its prompt cache key (the `prompt_cache_key` option, else
  the LLM conversation ID) as the `session-id` header, merged into any
  `extra_headers`, and as `prompt_cache_key`. The backend takes the key from
  the header and ignores a body-only key; without the header it assigns a
  random key per request and no request gets cache hits. Without a
  conversation and without the option (`model.prompt()`, `Response.reply()`),
  no key is sent; this is intentional for now and documented in `README.md`.
  The option accepts only valid header values (printable ASCII).
- `usage.attribution` (a per-item token breakdown that grows with the
  conversation) is removed from `response_json` and from the token details.
- Entries in `DEFAULT_MODELS` are raw API slugs without `codex/`; registration
  adds the prefix.
- Registration combines models discovered from the Codex API with hardcoded
  fallbacks and removes duplicates while preserving order.
- A fallback model can be listed even when the current plan cannot invoke it;
  preserve clear request-time errors for unavailable models.

## Authentication invariants

- Plugin-owned `auth-codex.json` takes precedence over borrowed Codex CLI auth.
- Borrowed auth is used only when plugin-owned auth does not exist.
- Import copies Codex CLI tokens into plugin-owned storage; it is not a new OAuth
  authorization.
- SCP imports copy only the fixed remote `~/.codex/auth.json`, discard the
  refresh token, and remain non-refreshable snapshots. Never replace existing
  plugin auth unless it has no refresh token and its access-token JWT has a
  decodable, expired expiry; perform that check before invoking `scp` and
  preserve existing auth on every transfer or validation failure.
- Refresh rotates the refresh token. A borrowed refresh writes the shared Codex
  CLI auth file and must retain its user-facing warning.
- Logout deletes local plugin-owned auth only; it does not revoke tokens or
  modify borrowed or remote auth.
- Never expose complete tokens in output, logs, fixtures, or test failures.
- Tests must use temporary auth paths and mocked network calls, never real user
  credentials or live OAuth flows.
- Preserve private permissions when writing plugin-owned authentication.

## User-facing changes

- Update `README.md` when commands, model behavior, authentication, options, or
  installation steps change.
- Keep examples under the public `codex/` model prefix.
- Keep release instructions consistent with `pyproject.toml` and the publishing
  workflow.

## Releasing

- Finish and verify feature/fix commits before changing the version.
- Before the version bump, run `uv lock --upgrade && uv run pytest`, so local
  tests use the newest dependency versions that `pyproject.toml` allows, as CI
  does. `uv.lock` stays untracked.
- Complete the whole flow: separate `bump: vX.Y.Z` commit changing
  `pyproject.toml` (its body starts the release notes), lightweight tag on that
  commit, then `git push origin main vX.Y.Z`. The tag push starts
  `publish.yml`, which tests, publishes to PyPI and creates the GitHub
  release. Do not run `gh release create`.
- Keep the PyPI publication and the release in `publish.yml`: a release made
  with the workflow token starts no other workflow. Do not rename the file;
  the PyPI trusted publisher is bound to it.
