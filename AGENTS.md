# Repository guidance

## Scope and layout

- `llm_openai_codex.py` contains the plugin implementation, model registration,
  Responses API integration, authentication, model discovery, usage reporting,
  and `llm codex` commands.
- `tests/test_openai_codex.py` is the main test suite.
- `README.md` is the user-facing source of truth for installation and behavior.

## Development

- Use `uv` for the development environment.
- Run the full suite with `uv run pytest` after code changes.
- Add or update tests for behavior changes, including error and fallback paths.
- Keep synchronous and asynchronous model behavior aligned through the shared
  implementation.

## Model invariants

- Public LLM model IDs use the `codex/` prefix.
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
- Put the `pyproject.toml` version bump in a separate `bump: vX.Y.Z` commit.
- Tag the bump commit, then push the branch and tag before creating the release.
