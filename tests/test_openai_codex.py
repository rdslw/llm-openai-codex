import base64
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
import llm
import pytest

from llm_openai_codex import (
    AUTH_MISSING_MESSAGE,
    AUTH_RECOVERY_MESSAGE,
    BorrowKeyError,
    CodexResponsesModel,
    DEFAULT_MODELS,
    DEVICE_REDIRECT_URI,
    DEVICE_TOKEN_URL,
    DEVICE_USER_CODE_URL,
    CHATGPT_BACKEND_BASE_URL,
    MODELS_CACHE_TTL,
    _account_id_from_token,
    _account_info_from_tokens,
    _auth_path,
    _browser_login,
    _cached_codex_models,
    _codex_cli_auth_path,
    _device_code_login,
    _ensure_account_id,
    _exchange_authorization_code,
    _fetch_codex_models,
    _import_codex_auth,
    _model_names_for_registration,
    _models_cache_path,
    _fetch_usage,
    _format_usage,
    _post_json_status,
    _read_auth,
    _refresh,
    _refresh_auth,
    _request_json,
    _resolve_auth,
    _write_auth,
    get_codex_key,
    codex,
)


def jwt(payload):
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    path = tmp_path / "llm-user-dir"
    monkeypatch.setenv("LLM_USER_PATH", str(path))
    return path


@pytest.fixture
def auth_file(tmp_path, monkeypatch):
    path = tmp_path / "auth-codex.json"
    monkeypatch.setenv("LLM_OPENAI_CODEX_AUTH_FILE", str(path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    return path


def write_codex_cli_auth(tmp_path, monkeypatch, data):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(exist_ok=True)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    path = codex_home / "auth.json"
    path.write_text(json.dumps(data))
    return path


def test_model_metadata():
    model = CodexResponsesModel("gpt-5.4")
    assert model.model_id == "codex/gpt-5.4"
    assert model.model_name == "gpt-5.4"
    assert str(model) == "OpenAI Codex: codex/gpt-5.4"
    assert model.needs_key is None
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


def test_build_kwargs_web_search():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(
        web_search=True, web_search_live=True, web_search_context_size="high"
    )
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["tools"] == [
        {
            "type": "web_search",
            "external_web_access": True,
            "search_context_size": "high",
        }
    ]


def test_build_kwargs_web_search_minimal():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(web_search=True)
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["tools"] == [{"type": "web_search"}]


def test_build_kwargs_no_web_search_by_default():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    kwargs = model._build_kwargs(prompt, None)
    assert "tools" not in kwargs


def test_build_kwargs_web_search_coexists_with_function_tool():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(web_search=True)
    prompt.tools = [llm.Tool(name="my_tool", description="d")]
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["tools"][0] == {"type": "web_search"}
    assert kwargs["tools"][1]["type"] == "function"
    assert kwargs["tools"][1]["name"] == "my_tool"


class FakePrevResponse:
    """Stand-in for a logged llm Response in conversation history."""

    def __init__(
        self, prompt_text, text="", tool_calls=None, tool_results=None, attachments=None
    ):
        self.prompt = SimpleNamespace(
            prompt=prompt_text, tool_results=tool_results or []
        )
        self.attachments = attachments or []
        self._text = text
        self._tool_calls = tool_calls or []

    def text_or_raise(self):
        return self._text

    def tool_calls_or_raise(self):
        return self._tool_calls


def test_build_messages_skips_empty_user_turn_with_tool_results():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="")
    prompt.tool_results = [
        llm.ToolResult(name="get_time", output="noon", tool_call_id="call_1")
    ]
    messages = model._build_messages(prompt, None)
    assert messages == [
        {"type": "function_call_output", "call_id": "call_1", "output": "noon"}
    ]


def test_build_messages_replays_tool_loop_history_without_empty_user_turns():
    model = CodexResponsesModel("gpt-5.4")
    conversation = SimpleNamespace(
        responses=[
            FakePrevResponse(
                "What time?",
                tool_calls=[
                    llm.ToolCall(
                        tool_call_id="call_1", name="get_time", arguments={}
                    )
                ],
            ),
            FakePrevResponse(
                None,
                text="It is noon.",
                tool_results=[
                    llm.ToolResult(
                        name="get_time", output="noon", tool_call_id="call_1"
                    )
                ],
            ),
        ]
    )
    prompt = llm.Prompt(model=model, prompt="Thanks!")
    messages = model._build_messages(prompt, conversation)
    assert messages == [
        {"role": "user", "content": "What time?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_time",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "noon"},
        {"role": "assistant", "content": "It is noon."},
        {"role": "user", "content": "Thanks!"},
    ]


class RecordingResponse:
    """Stand-in for llm's Response in _handle_event tests."""

    def __init__(self):
        self.response_json = None
        self.usage = None
        self.tool_calls = []

    def set_usage(self, input=None, output=None, details=None):
        self.usage = (input, output, details)

    def add_tool_call(self, tool_call):
        self.tool_calls.append(tool_call)


def test_build_messages_with_attachments_in_history_and_prompt():
    model = CodexResponsesModel("gpt-5.4")
    image = SimpleNamespace(url="https://example.com/old.png")
    conversation = SimpleNamespace(
        responses=[
            FakePrevResponse("Look at this", text="Nice.", attachments=[image])
        ]
    )
    prompt = llm.Prompt(model=model, prompt="And this?")
    prompt.attachments = [SimpleNamespace(url="https://example.com/new.png")]
    messages = model._build_messages(prompt, conversation)
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look at this"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/old.png",
                    "detail": "low",
                },
            ],
        },
        {"role": "assistant", "content": "Nice."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "And this?"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/new.png",
                    "detail": "low",
                },
            ],
        },
    ]


def test_handle_event_output_text_delta_returns_delta():
    model = CodexResponsesModel("gpt-5.4")
    event = SimpleNamespace(type="response.output_text.delta", delta="Hel")
    assert model._handle_event(event, RecordingResponse()) == "Hel"


def test_handle_event_function_call_adds_tool_call():
    model = CodexResponsesModel("gpt-5.4")
    event = SimpleNamespace(
        type="response.output_item.done",
        item={
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_time",
            "arguments": '{"tz": "UTC"}',
        },
    )
    response = RecordingResponse()
    assert model._handle_event(event, response) is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_call_id == "call_1"
    assert response.tool_calls[0].name == "get_time"
    assert response.tool_calls[0].arguments == {"tz": "UTC"}


def test_handle_event_completed_sets_response_json_and_usage():
    model = CodexResponsesModel("gpt-5.4")
    event = SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            model_dump=lambda: {"status": "completed"},
        ),
    )
    response = RecordingResponse()
    assert model._handle_event(event, response) is None
    assert response.response_json == {"status": "completed"}
    assert response.usage == (7, 3, {})


def test_set_usage_tolerates_missing_token_fields():
    model = CodexResponsesModel("gpt-5.4")
    response = RecordingResponse()
    model.set_usage(response, {"output_tokens": 5})
    assert response.usage == (None, 5, {})


def test_read_auth_reports_invalid_json(auth_file):
    auth_file.write_text("{not json")
    with pytest.raises(BorrowKeyError, match="Invalid JSON in auth file"):
        _read_auth(auth_file)


def test_handle_event_failed_raises_model_error():
    model = CodexResponsesModel("gpt-5.4")
    event = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            usage=None,
            error=SimpleNamespace(message="quota exceeded"),
            model_dump=lambda: {"status": "failed"},
        ),
    )
    response = RecordingResponse()
    with pytest.raises(llm.ModelError, match="quota exceeded"):
        model._handle_event(event, response)
    assert response.response_json == {"status": "failed"}


def test_handle_event_incomplete_records_response_and_usage():
    model = CodexResponsesModel("gpt-5.4")
    event = SimpleNamespace(
        type="response.incomplete",
        response=SimpleNamespace(
            usage={"input_tokens": 1, "output_tokens": 2},
            model_dump=lambda: {"status": "incomplete"},
        ),
    )
    response = RecordingResponse()
    assert model._handle_event(event, response) is None
    assert response.response_json == {"status": "incomplete"}
    assert response.usage == (1, 2, {})


def test_handle_event_error_event_raises_model_error():
    model = CodexResponsesModel("gpt-5.4")
    event = SimpleNamespace(type="error", message="bad request", code="server_error")
    with pytest.raises(llm.ModelError, match="bad request"):
        model._handle_event(event, RecordingResponse())


def test_fetch_codex_models_returns_none_without_auth():
    with patch(
        "llm_openai_codex.get_codex_key",
        side_effect=BorrowKeyError("no auth"),
    ):
        models = _fetch_codex_models()
    assert models is None


def test_model_registration_falls_back_to_defaults():
    with patch("llm_openai_codex._fetch_codex_models", return_value=None):
        models = _model_names_for_registration()
    assert models == DEFAULT_MODELS


def test_cached_models_skip_fetch_while_fresh():
    _models_cache_path().write_text(
        json.dumps({"fetched_at": time.time(), "models": ["gpt-cached"]})
    )
    with patch("llm_openai_codex._fetch_codex_models") as fetch:
        models = _cached_codex_models()
    fetch.assert_not_called()
    assert models == ["gpt-cached"]


def test_stale_cache_is_refetched_and_rewritten():
    _models_cache_path().write_text(
        json.dumps(
            {
                "fetched_at": time.time() - MODELS_CACHE_TTL - 1,
                "models": ["gpt-stale"],
            }
        )
    )
    with patch("llm_openai_codex._fetch_codex_models", return_value=["gpt-fresh"]):
        models = _cached_codex_models()
    assert models == ["gpt-fresh"]
    assert json.loads(_models_cache_path().read_text())["models"] == ["gpt-fresh"]


def test_stale_cache_beats_failed_fetch():
    _models_cache_path().write_text(
        json.dumps(
            {
                "fetched_at": time.time() - MODELS_CACHE_TTL - 1,
                "models": ["gpt-stale"],
            }
        )
    )
    with patch("llm_openai_codex._fetch_codex_models", return_value=None):
        models = _cached_codex_models()
    assert models == ["gpt-stale"]


def test_fallback_models_include_codex_spark():
    assert "gpt-5.3-codex-spark" in DEFAULT_MODELS


def test_model_registration_includes_defaults_when_discovery_succeeds():
    with patch("llm_openai_codex._fetch_codex_models", return_value=["gpt-test"]):
        models = _model_names_for_registration()
    assert models[0] == "gpt-test"
    assert "gpt-5.3-codex-spark" in models


def test_fetch_codex_models_suppresses_default_user_agent():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-test",
                            "supported_in_api": True,
                            "visibility": "list",
                        }
                    ]
                }
            ).encode()

    captured = {}

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("llm_openai_codex.get_codex_key", return_value=("token", "acct")):
        with patch("llm_openai_codex.urllib.request.urlopen", fake_urlopen):
            models = _fetch_codex_models()

    assert models == ["gpt-test"]
    assert captured["timeout"] == 10
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["headers"]["Chatgpt-account-id"] == "acct"
    assert captured["headers"]["User-agent"] == ""


def test_request_json_sets_timeout():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"ok": true}'

    captured = {}

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("llm_openai_codex.urllib.request.urlopen", fake_urlopen):
        data = _request_json("https://example.com", {})

    assert data == {"ok": True}
    assert captured["timeout"] == 20


def test_fetch_usage_uses_wham_usage_endpoint_and_auth_headers():
    captured = {}

    def fake_request(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return {"plan_type": "plus"}

    with patch("llm_openai_codex.get_codex_key", return_value=("token", "acct")):
        with patch("llm_openai_codex._request_json", fake_request):
            payload = _fetch_usage()

    assert payload == {"plan_type": "plus"}
    assert captured["url"] == f"{CHATGPT_BACKEND_BASE_URL}/wham/usage"
    assert captured["headers"] == {
        "Authorization": "Bearer token",
        "User-Agent": "",
        "ChatGPT-Account-ID": "acct",
    }


def test_format_usage_shows_limits_and_credits():
    payload = {
        "plan_type": "plus",
        "account_email": "user@example.com",
        "rate_limit": {
            "primary_window": {
                "used_percent": 74,
                "limit_window_seconds": 18000,
                "reset_at": 1777210200,
            },
            "secondary_window": {
                "used_percent": 24,
                "limit_window_seconds": 604800,
                "reset_at": 1777464540,
            },
        },
        "credits": {
            "has_credits": True,
            "unlimited": False,
            "balance": "12.4",
        },
    }
    berlin = timezone(timedelta(hours=2))
    now = datetime(2026, 4, 26, 14, 0, tzinfo=berlin)
    output = _format_usage(payload, now=now)

    assert output.startswith(
        "Codex usage details: https://chatgpt.com/codex/settings/usage"
    )
    assert "Account: user@example.com (Plus)" in output
    assert "5h limit: [█████░░░░░░░░░░░░░░░] 26% left (resets 15:30)" in output
    assert (
        "Weekly limit: [███████████████░░░░░] 76% left "
        "(resets 14:09 on 29 Apr)"
    ) in output
    assert "Credits: 12 credits" in output


def test_format_usage_shows_unlimited_credits():
    payload = {
        "rate_limit": None,
        "credits": {"has_credits": True, "unlimited": True, "balance": None},
    }

    output = _format_usage(payload)

    assert "Credits: Unlimited" in output


def test_format_usage_omits_plan_without_account_email():
    output = _format_usage({"plan_type": "plus"})

    assert "Plan:" not in output
    assert "Account:" not in output
    assert "No usage limit data returned." in output


def test_format_usage_shows_rate_limit_reached_type():
    output = _format_usage(
        {
            "rate_limit_reached_type": "workspace_member_usage_limit_reached",
            "rate_limit": {
                "allowed": False,
                "limit_reached": True,
                "primary_window": {
                    "used_percent": 100,
                    "limit_window_seconds": 18000,
                    "reset_at": None,
                },
            },
        }
    )

    assert "Rate limit: Workspace member usage limit reached" in output
    assert "5h limit: [░░░░░░░░░░░░░░░░░░░░] 0% left" in output


def test_format_usage_shows_limit_reached_without_reached_type():
    output = _format_usage(
        {
            "rate_limit": {
                "allowed": False,
                "limit_reached": True,
            },
        }
    )

    assert "Rate limit: Rate limit reached" in output


def test_usage_command_prints_formatted_usage():
    with patch(
        "llm_openai_codex._fetch_usage",
        return_value={"plan_type": "plus", "account_email": "user@example.com"},
    ):
        result = CliRunner().invoke(codex, ["usage"])

    assert result.exit_code == 0
    assert (
        "Codex usage details: https://chatgpt.com/codex/settings/usage"
        in result.output
    )
    assert "Account: user@example.com (Plus)" in result.output


def test_write_auth_creates_private_file(auth_file):
    _write_auth(
        auth_file,
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "access"},
        },
    )
    assert auth_file.stat().st_mode & 0o777 == 0o600


def test_auth_path_uses_override(auth_file):
    assert _auth_path() == auth_file


def test_codex_cli_auth_path_uses_codex_home(auth_file, tmp_path, monkeypatch):
    codex_home = tmp_path / "custom-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    assert _codex_cli_auth_path() == codex_home / "auth.json"


def test_resolve_auth_uses_plugin_auth_first(auth_file, tmp_path, monkeypatch):
    plugin_data = {
        "auth_mode": "chatgpt",
        "tokens": {"access_token": "plugin"},
    }
    _write_auth(auth_file, plugin_data)
    write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {"auth_mode": "chatgpt", "tokens": {"access_token": "cli"}},
    )

    auth = _resolve_auth()

    assert auth.path == auth_file
    assert auth.label == "plugin-owned auth"
    assert auth.data["tokens"]["access_token"] == "plugin"
    assert auth.read_only is False
    assert auth.refreshable is True


def test_resolve_auth_falls_back_to_codex_cli_auth(auth_file, tmp_path, monkeypatch):
    cli_path = write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {"auth_mode": "chatgpt", "tokens": {"access_token": "cli"}},
    )

    auth = _resolve_auth()

    assert auth.path == cli_path
    assert auth.label == "Codex CLI borrowed auth"
    assert auth.data["tokens"]["access_token"] == "cli"
    assert auth.read_only is True
    assert auth.refreshable is False


def test_existing_invalid_plugin_auth_does_not_fall_back_to_codex_cli(
    auth_file, tmp_path, monkeypatch
):
    _write_auth(auth_file, {"auth_mode": "api", "tokens": {"access_token": "plugin"}})
    write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {"auth_mode": "chatgpt", "tokens": {"access_token": "cli"}},
    )

    with pytest.raises(BorrowKeyError, match="Expected auth_mode 'chatgpt'"):
        _resolve_auth()


def test_get_codex_key_uses_codex_cli_fallback_without_persisting(
    auth_file, tmp_path, monkeypatch
):
    token = jwt(
        {
            "exp": int(time.time()) + 3600,
            "chatgpt_account_id": "acct_from_access",
        }
    )
    cli_path = write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {"auth_mode": "chatgpt", "tokens": {"access_token": token}},
    )

    assert get_codex_key() == (token, "acct_from_access")
    saved = json.loads(cli_path.read_text())
    assert "account_id" not in saved["tokens"]


def test_get_codex_key_refuses_to_refresh_codex_cli_fallback(
    auth_file, tmp_path, monkeypatch
):
    token = jwt({"exp": int(time.time()) - 10})
    write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": token, "refresh_token": "refresh"},
        },
    )

    with patch("llm_openai_codex._refresh") as refresh:
        with pytest.raises(BorrowKeyError) as excinfo:
            get_codex_key()

    refresh.assert_not_called()
    assert "Borrowed Codex CLI auth token is expired" in str(excinfo.value)
    assert "llm codex refresh --borrowed" in str(excinfo.value)
    assert AUTH_RECOVERY_MESSAGE in str(excinfo.value)


def test_get_codex_key_reports_missing_plugin_access_token(auth_file):
    _write_auth(auth_file, {"auth_mode": "chatgpt", "tokens": {"refresh_token": "r"}})

    with pytest.raises(BorrowKeyError) as excinfo:
        get_codex_key()

    assert "Plugin-owned auth" in str(excinfo.value)
    assert "does not contain an access token" in str(excinfo.value)
    assert AUTH_RECOVERY_MESSAGE in str(excinfo.value)


def test_get_codex_key_reports_missing_codex_cli_access_token(
    auth_file, tmp_path, monkeypatch
):
    write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {"auth_mode": "chatgpt", "tokens": {"refresh_token": "r"}},
    )

    with pytest.raises(BorrowKeyError) as excinfo:
        get_codex_key()

    assert "Codex CLI auth" in str(excinfo.value)
    assert "does not contain an access token" in str(excinfo.value)
    assert "llm codex refresh --borrowed" in str(excinfo.value)


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


def test_account_info_from_tokens_extracts_email_and_plan():
    # access_token: email in namespaced profile claim, plan in auth claim
    access = jwt({
        "https://api.openai.com/profile": {"email": "a@example.com"},
        "https://api.openai.com/auth": {"chatgpt_plan_type": "plus"},
    })
    # id_token: top-level email; preferred only when access_token has nothing
    id_tok = jwt({"email": "fallback@example.com",
                  "https://api.openai.com/auth": {"chatgpt_plan_type": "pro"}})

    assert _account_info_from_tokens({"access_token": access, "id_token": id_tok}) == (
        "a@example.com", "plus",
    )
    # Falls back to id_token when access_token lacks claims
    assert _account_info_from_tokens({"access_token": jwt({}), "id_token": id_tok}) == (
        "fallback@example.com", "pro",
    )
    # No tokens / no claims / non-dict auth claim
    assert _account_info_from_tokens({}) == (None, None)
    assert _account_info_from_tokens({"access_token": jwt({"https://api.openai.com/auth": "junk"})}) == (None, None)


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
    assert AUTH_RECOVERY_MESSAGE in result.output
    assert "Plugin auth path:" in result.output
    assert "Codex CLI auth path:" in result.output


def test_status_shows_plugin_auth_source(auth_file):
    _write_auth(
        auth_file,
        {"auth_mode": "chatgpt", "login_type": "chatgpt", "tokens": {"access_token": "x"}},
    )

    result = CliRunner().invoke(codex, ["status"])

    assert result.exit_code == 0
    assert "Auth source: plugin-owned auth" in result.output
    assert f"Auth file: {auth_file}" in result.output


def test_status_shows_codex_cli_fallback_source(auth_file, tmp_path, monkeypatch):
    cli_path = write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {
            "auth_mode": "chatgpt",
            "login_type": "chatgpt",
            "tokens": {
                "access_token": "access",
                "id_token": jwt({"chatgpt_account_id": "acct_cli"}),
            },
        },
    )

    result = CliRunner().invoke(codex, ["status"])

    assert result.exit_code == 0
    assert "Auth source: Codex CLI borrowed auth" in result.output
    assert f"Auth file: {cli_path}" in result.output
    assert "account_id: acct_cli" in result.output


def test_missing_refresh_token_uses_common_auth_recovery_message(auth_file):
    data = {"auth_mode": "chatgpt", "tokens": {"access_token": "expired"}}
    with pytest.raises(BorrowKeyError) as excinfo:
        _refresh_auth(data, auth_file)
    assert AUTH_RECOVERY_MESSAGE in str(excinfo.value)


def test_invalid_refresh_token_uses_common_auth_recovery_message():
    with patch(
        "llm_openai_codex._post_json_status",
        return_value=(400, {"error": "refresh_token_expired"}),
    ):
        with pytest.raises(BorrowKeyError) as excinfo:
            _refresh("refresh")
    assert AUTH_RECOVERY_MESSAGE in str(excinfo.value)


def test_logout_removes_file(auth_file):
    _write_auth(auth_file, {"auth_mode": "chatgpt", "tokens": {"access_token": "x"}})
    result = CliRunner().invoke(codex, ["logout"])
    assert result.exit_code == 0
    assert not auth_file.exists()


def test_logout_is_disabled_for_codex_cli_fallback(auth_file, tmp_path, monkeypatch):
    cli_path = write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {"auth_mode": "chatgpt", "tokens": {"access_token": "cli"}},
    )

    result = CliRunner().invoke(codex, ["logout"])

    assert result.exit_code != 0
    assert "Cannot logout from borrowed Codex CLI auth" in result.output
    assert str(cli_path) in result.output
    assert cli_path.exists()


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


def test_import_command_refuses_to_overwrite_existing_plugin_auth(auth_file, tmp_path):
    _write_auth(auth_file, {"auth_mode": "chatgpt", "tokens": {"access_token": "plugin"}})
    source = tmp_path / "auth.json"
    source.write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "cli"}})
    )

    result = CliRunner().invoke(codex, ["import", "--path", str(source)])

    assert result.exit_code != 0
    assert "Plugin-owned auth already exists" in result.output
    assert json.loads(auth_file.read_text())["tokens"]["access_token"] == "plugin"


def test_import_command_reports_missing_codex_cli_auth(auth_file, tmp_path):
    source = tmp_path / "missing-auth.json"

    result = CliRunner().invoke(codex, ["import", "--path", str(source)])

    assert result.exit_code != 0
    assert f"No Codex CLI auth found at {source}" in result.output
    assert AUTH_RECOVERY_MESSAGE in result.output


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


def test_refresh_command_requires_borrowed_flag_for_codex_cli_fallback(
    auth_file, tmp_path, monkeypatch
):
    cli_path = write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "cli", "refresh_token": "refresh"},
        },
    )

    with patch("llm_openai_codex._refresh") as refresh:
        result = CliRunner().invoke(codex, ["refresh"])

    assert result.exit_code != 0
    assert "requires explicit opt-in" in result.output
    assert "--borrowed" in result.output
    assert "never auto-refreshed" in result.output
    assert str(cli_path) in result.output
    refresh.assert_not_called()


def test_refresh_command_borrowed_flag_refreshes_codex_cli_fallback(
    auth_file, tmp_path, monkeypatch
):
    cli_path = write_codex_cli_auth(
        tmp_path,
        monkeypatch,
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "cli", "refresh_token": "refresh"},
        },
    )

    with patch(
        "llm_openai_codex._refresh",
        return_value={
            "access_token": "new_cli",
            "id_token": jwt({"chatgpt_account_id": "acct_cli"}),
            "refresh_token": "rotated_refresh",
        },
    ):
        result = CliRunner().invoke(codex, ["refresh", "--borrowed"])

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "rotates the shared refresh_token" in result.output
    assert "restart any running Codex CLI sessions" in result.output
    assert f"Refreshed Codex auth at {cli_path}" in result.output
    saved = json.loads(cli_path.read_text())
    assert saved["tokens"]["access_token"] == "new_cli"
    assert saved["tokens"]["refresh_token"] == "rotated_refresh"


def test_refresh_command_borrowed_flag_is_ignored_for_plugin_owned(auth_file):
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
        result = CliRunner().invoke(codex, ["refresh", "--borrowed"])
    assert result.exit_code == 0, result.output
    assert "--borrowed ignored" in result.output
    assert json.loads(auth_file.read_text())["tokens"]["access_token"] == "new"


def test_browser_login_survives_stray_requests(monkeypatch):
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request

    opened = {}
    monkeypatch.setattr(
        "llm_openai_codex.webbrowser.open", lambda url: opened.setdefault("url", url)
    )
    exchanged = {}

    def fake_exchange(code, code_verifier, redirect_uri=None):
        exchanged["code"] = code
        return {"access_token": "access"}

    monkeypatch.setattr(
        "llm_openai_codex._exchange_authorization_code", fake_exchange
    )

    result = {}

    def run():
        result["tokens"] = _browser_login()

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(200):
        if "url" in opened:
            break
        time.sleep(0.05)
    state = urllib.parse.parse_qs(urllib.parse.urlparse(opened["url"]).query)["state"][0]

    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen("http://127.0.0.1:1455/favicon.ico", timeout=5)
    urllib.request.urlopen(
        f"http://127.0.0.1:1455/auth/callback?code=abc&state={state}", timeout=5
    )

    thread.join(timeout=10)
    assert not thread.is_alive()
    assert result["tokens"] == {"access_token": "access"}
    assert exchanged["code"] == "abc"


def test_device_code_login_matches_codex_flow(capsys):
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
    output = capsys.readouterr().out
    assert "enabled device code authorization for Codex" in output
    assert output.index("enabled device code authorization for Codex") < output.index(
        "Enter code: CODE-12345"
    )
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


def test_post_json_status_uses_urllib_without_default_user_agent():
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"ok": true}'

    captured = {}

    def fake_urlopen(req, timeout):
        captured["request"] = req
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("llm_openai_codex.urllib.request.urlopen", fake_urlopen):
        status, data = _post_json_status("https://example.com/path", {"x": 1})

    assert status == 200
    assert data == {"ok": True}
    assert captured["timeout"] == 20
    assert captured["request"].full_url == "https://example.com/path"
    assert captured["request"].data == b'{"x": 1}'
    assert dict(captured["request"].header_items()) == {
        "Accept": "application/json",
        "Content-type": "application/json",
        "User-agent": "",
    }


def test_refresh_and_exchange_use_json_post_helper():
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return 200, {"access_token": "access"}

    with patch("llm_openai_codex._post_json_status", fake_post):
        assert _refresh("refresh") == {"access_token": "access"}
        assert _exchange_authorization_code("code", "verifier") == {
            "access_token": "access"
        }

    assert calls[0][1] == {
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "grant_type": "refresh_token",
        "refresh_token": "refresh",
    }
    assert calls[1][1] == {
        "grant_type": "authorization_code",
        "code": "code",
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "code_verifier": "verifier",
        "redirect_uri": "http://localhost:1455/auth/callback",
    }
