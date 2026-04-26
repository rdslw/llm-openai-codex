import base64
import json
import stat
import time
from unittest.mock import patch

from click.testing import CliRunner
import llm
import pytest

from llm_openai_codex import (
    AUTH_MISSING_MESSAGE,
    BorrowKeyError,
    CodexResponsesModel,
    DEFAULT_MODELS,
    DEVICE_REDIRECT_URI,
    DEVICE_TOKEN_URL,
    DEVICE_USER_CODE_URL,
    _account_id_from_token,
    _auth_path,
    _device_code_login,
    _ensure_account_id,
    _fetch_codex_models,
    _import_codex_auth,
    _read_auth,
    _refresh_auth,
    _write_auth,
    codex,
)


def jwt(payload):
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


@pytest.fixture
def auth_file(tmp_path, monkeypatch):
    path = tmp_path / "auth-codex.json"
    monkeypatch.setenv("LLM_OPENAI_CODEX_AUTH_FILE", str(path))
    return path


def test_plugin_is_installed():
    import llm_openai_codex


def test_models_are_registered():
    model_ids = [model.model_id for model in llm.get_models()]
    # At least the default models should be registered (or fetched ones)
    # Check the prefix is correct
    codex_models = [m for m in model_ids if m.startswith("codex/")]
    assert len(codex_models) > 0


def test_model_id_prefix():
    model = CodexResponsesModel("gpt-5.4")
    assert model.model_id == "codex/gpt-5.4"
    assert model.model_name == "gpt-5.4"
    assert str(model) == "OpenAI Codex: codex/gpt-5.4"


def test_model_needs_no_key():
    model = CodexResponsesModel("gpt-5.4")
    assert model.needs_key is None


def test_model_can_stream():
    model = CodexResponsesModel("gpt-5.4")
    assert model.can_stream is True


def test_build_kwargs_basic():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["model"] == "gpt-5.4"
    assert kwargs["store"] is False
    assert kwargs["stream"] is True
    assert kwargs["instructions"] == "You are a helpful assistant."
    assert kwargs["input"] == [{"role": "user", "content": "Hello"}]


def test_build_kwargs_with_system():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello", system="Be brief.")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["instructions"] == "Be brief."


def test_build_kwargs_with_options():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(temperature=0.5, max_output_tokens=100, top_p=0.9)
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_output_tokens"] == 100
    assert kwargs["top_p"] == 0.9


def test_build_kwargs_reasoning_effort():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(reasoning_effort="high")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["reasoning"] == {"effort": "high"}


def test_build_kwargs_verbosity():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(verbosity="low")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["text"]["verbosity"] == "low"


def test_build_kwargs_verbosity_and_schema():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(verbosity="high")
    prompt.schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["text"]["verbosity"] == "high"
    assert kwargs["text"]["format"]["schema"] == prompt.schema


def test_build_kwargs_forwards_extra_options():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(service_tier="flex")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["service_tier"] == "flex"


def test_fetch_codex_models_fallback():
    with patch(
        "llm_openai_codex.get_codex_key",
        side_effect=BorrowKeyError("no auth"),
    ):
        models = _fetch_codex_models()
    assert models == DEFAULT_MODELS


def test_write_auth_creates_private_file(auth_file):
    _write_auth(
        auth_file,
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "access"},
        },
    )
    assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600


def test_auth_path_uses_override(auth_file):
    assert _auth_path() == auth_file


def test_account_id_from_token_claim_order():
    assert _account_id_from_token(jwt({"chatgpt_account_id": "acct_top"})) == "acct_top"
    assert (
        _account_id_from_token(
            jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_nested"}})
        )
        == "acct_nested"
    )
    assert _account_id_from_token(jwt({"organizations": [{"id": "org_1"}]})) == "org_1"
    assert _account_id_from_token(jwt({"organization_id": "org_2"})) == "org_2"


def test_missing_account_id_is_derived_and_persisted(auth_file):
    data = {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "access",
            "id_token": jwt({"chatgpt_account_id": "acct_from_id"}),
        },
    }
    _write_auth(auth_file, data)
    loaded = _read_auth(auth_file)
    assert _ensure_account_id(loaded, persist_path=auth_file) == "acct_from_id"
    assert json.loads(auth_file.read_text())["tokens"]["account_id"] == "acct_from_id"


def test_existing_account_id_is_preserved(auth_file):
    data = {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "access",
            "id_token": jwt({"chatgpt_account_id": "acct_from_id"}),
            "account_id": "acct_existing",
        },
    }
    assert _ensure_account_id(data, persist_path=auth_file) == "acct_existing"
    assert data["tokens"]["account_id"] == "acct_existing"


def test_import_copies_codex_auth(auth_file, tmp_path):
    source = tmp_path / "auth.json"
    source.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "id_token": jwt({"chatgpt_account_id": "acct_imported"}),
                },
            }
        )
    )
    path, data = _import_codex_auth(source)
    assert path == auth_file
    assert data["login_type"] == "import"
    assert data["tokens"]["account_id"] == "acct_imported"
    assert json.loads(auth_file.read_text())["tokens"]["refresh_token"] == "refresh"


def test_refresh_persists_updates(auth_file):
    data = {
        "auth_mode": "chatgpt",
        "tokens": {"access_token": "old", "refresh_token": "refresh"},
    }
    with patch(
        "llm_openai_codex._refresh",
        return_value={
            "access_token": jwt({"exp": int(time.time()) + 3600}),
            "id_token": jwt({"chatgpt_account_id": "acct_refreshed"}),
            "refresh_token": "new_refresh",
        },
    ):
        _refresh_auth(data, auth_file)
    saved = json.loads(auth_file.read_text())
    assert saved["tokens"]["refresh_token"] == "new_refresh"
    assert saved["tokens"]["account_id"] == "acct_refreshed"
    assert saved["last_refresh"]


def test_status_missing_auth_exits_cleanly(auth_file):
    result = CliRunner().invoke(codex, ["status"])
    assert result.exit_code == 0
    assert AUTH_MISSING_MESSAGE in result.output


def test_logout_removes_file(auth_file):
    _write_auth(auth_file, {"auth_mode": "chatgpt", "tokens": {"access_token": "x"}})
    result = CliRunner().invoke(codex, ["logout"])
    assert result.exit_code == 0
    assert not auth_file.exists()


def test_import_command_copies_auth(auth_file, tmp_path):
    source = tmp_path / "auth.json"
    source.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "access",
                    "id_token": jwt({"chatgpt_account_id": "acct_cli"}),
                },
            }
        )
    )
    result = CliRunner().invoke(codex, ["import", "--path", str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(auth_file.read_text())["tokens"]["account_id"] == "acct_cli"


def test_refresh_command_persists_updates(auth_file):
    _write_auth(
        auth_file,
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "old", "refresh_token": "refresh"},
        },
    )
    with patch(
        "llm_openai_codex._refresh",
        return_value={"access_token": "new", "id_token": jwt({"chatgpt_account_id": "a"})},
    ):
        result = CliRunner().invoke(codex, ["refresh"])
    assert result.exit_code == 0, result.output
    assert json.loads(auth_file.read_text())["tokens"]["access_token"] == "new"


def test_device_code_login_matches_codex_flow():
    responses = [
        (
            200,
            {
                "device_auth_id": "device-auth-123",
                "user_code": "CODE-12345",
                "interval": "0",
            },
        ),
        (
            200,
            {
                "authorization_code": "poll-code-321",
                "code_challenge": "code-challenge-321",
                "code_verifier": "code-verifier-321",
            },
        ),
    ]
    with patch("llm_openai_codex._post_json_status", side_effect=responses) as post_json:
        with patch(
            "llm_openai_codex._exchange_authorization_code",
            return_value={"access_token": "access"},
        ) as exchange:
            tokens = _device_code_login()

    assert tokens == {"access_token": "access"}
    assert post_json.call_args_list[0].args == (
        DEVICE_USER_CODE_URL,
        {"client_id": "app_EMoamEEZ73f0CkXaXp7hrann"},
    )
    assert post_json.call_args_list[1].args == (
        DEVICE_TOKEN_URL,
        {"device_auth_id": "device-auth-123", "user_code": "CODE-12345"},
    )
    exchange.assert_called_once_with(
        "poll-code-321",
        "code-verifier-321",
        redirect_uri=DEVICE_REDIRECT_URI,
    )


def test_device_code_login_reports_disabled_server():
    with patch("llm_openai_codex._post_json_status", return_value=(404, {})):
        with pytest.raises(BorrowKeyError, match="not enabled"):
            _device_code_login()
