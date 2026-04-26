import base64
import json
from unittest.mock import patch

import llm
import pytest

from llm_openai_codex import (
    BorrowKeyError,
    CodexResponsesModel,
    DEFAULT_MODELS,
    _account_id_from_token,
    _ensure_account_id,
    _fetch_codex_models,
    _read_auth,
    _write_auth,
)


def jwt(payload):
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


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
        "llm_openai_codex.borrow_codex_key",
        side_effect=BorrowKeyError("no auth"),
    ):
        models = _fetch_codex_models()
    assert models == DEFAULT_MODELS


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


def test_missing_account_id_is_derived_and_persisted(tmp_path):
    auth_file = tmp_path / "auth.json"
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


def test_existing_account_id_is_preserved(tmp_path):
    auth_file = tmp_path / "auth.json"
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
