import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import time
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from enum import Enum
from typing import Optional

import click
import llm
from llm import AsyncModel, Model, Options, hookimpl
from llm.utils import simplify_usage_dict
import openai
from pydantic import ConfigDict, Field


# --- Plugin-owned Codex auth ---

REFRESH_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 30
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
AUTH_MISSING_MESSAGE = (
    "No llm-openai-codex auth found. Run `llm codex login` or `llm codex import`."
)
REDIRECT_URI = "http://localhost:1455/auth/callback"
DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
DEVICE_VERIFICATION_URL = "https://auth.openai.com/codex/device"
DEVICE_USER_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"


class BorrowKeyError(Exception):
    pass


def get_codex_key():
    """
    Return (access_token, account_id) from plugin-owned ChatGPT OAuth
    credentials, refreshing the access token when it is expired or near-expiry.
    """
    auth_path = _auth_path()
    data = _read_auth(auth_path)

    tokens = data.get("tokens")
    if not tokens or not tokens.get("access_token"):
        raise BorrowKeyError(AUTH_MISSING_MESSAGE)

    _ensure_account_id(data, persist_path=auth_path)

    access_token = tokens["access_token"]
    account_id = tokens.get("account_id")
    exp = _jwt_exp(access_token)

    if exp is not None and time.time() < (exp - REFRESH_SKEW_SECONDS):
        return access_token, account_id

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise BorrowKeyError(
            "No refresh token available. Run `llm codex login` to re-authenticate."
        )

    _refresh_auth(data, auth_path)

    return tokens["access_token"], tokens.get("account_id")


# Backwards-compatible alias for callers that imported the old helper.
borrow_codex_key = get_codex_key


def _auth_path():
    override = os.environ.get("LLM_OPENAI_CODEX_AUTH_FILE")
    if override:
        return Path(override)
    return llm.user_dir() / "auth-codex.json"


def _codex_cli_auth_path():
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def _read_auth(path):
    path = Path(path)
    if not path.exists():
        raise BorrowKeyError(AUTH_MISSING_MESSAGE)
    with path.open() as f:
        data = json.load(f)
    if data.get("auth_mode") != "chatgpt":
        raise BorrowKeyError(
            f"Expected auth_mode 'chatgpt', got '{data.get('auth_mode')}'. "
            "This library only supports ChatGPT OAuth tokens."
        )
    return data


def _write_auth(path, data):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _jwt_payload(token):
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None


def _jwt_exp(token):
    payload = _jwt_payload(token)
    if payload:
        return payload.get("exp")
    return None


def _account_id_from_token(token):
    payload = _jwt_payload(token)
    if not payload:
        return None
    account_id = payload.get("chatgpt_account_id")
    if account_id:
        return account_id
    auth_claims = payload.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict) and auth_claims.get("chatgpt_account_id"):
        return auth_claims["chatgpt_account_id"]
    organizations = payload.get("organizations")
    if isinstance(organizations, list) and organizations:
        organization_id = organizations[0].get("id")
        if isinstance(organization_id, str) and organization_id:
            return organization_id
    organization_id = payload.get("organization_id")
    if isinstance(organization_id, str) and organization_id:
        return organization_id
    return None


def _ensure_account_id(data, persist_path=None):
    tokens = data.get("tokens") or {}
    if tokens.get("account_id"):
        return tokens["account_id"]
    account_id = None
    for token_key in ("id_token", "access_token"):
        token = tokens.get(token_key)
        if token:
            account_id = _account_id_from_token(token)
            if account_id:
                break
    if account_id:
        tokens["account_id"] = account_id
        data["tokens"] = tokens
        if persist_path is not None:
            _write_auth(persist_path, data)
    return account_id


def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_auth_data(tokens, login_type):
    data = {
        "auth_mode": "chatgpt",
        "login_type": login_type,
        "tokens": tokens,
        "last_refresh": _now_iso(),
    }
    _ensure_account_id(data)
    return data


def _refresh(refresh_token):
    body = json.dumps(
        {
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode()

    req = urllib.request.Request(
        REFRESH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        try:
            error_data = json.loads(error_body)
            error_code = error_data.get("error")
        except Exception:
            error_code = None

        if error_code in (
            "refresh_token_expired",
            "refresh_token_reused",
            "refresh_token_invalidated",
        ):
            raise BorrowKeyError(
                f"Refresh token is no longer valid ({error_code}). "
                "Run `llm codex login` to re-authenticate."
            ) from None

        raise BorrowKeyError(
            f"Token refresh failed (HTTP {e.code}): {error_body}"
        ) from None
    except urllib.error.URLError as e:
        raise BorrowKeyError(f"Token refresh failed (network error): {e}") from None


def _refresh_auth(data, auth_path=None):
    tokens = data.get("tokens") or {}
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise BorrowKeyError(
            "No refresh token available. Run `llm codex login` to re-authenticate."
        )
    new_tokens = _refresh(refresh_token)
    for token_key in ("access_token", "id_token", "refresh_token"):
        if new_tokens.get(token_key):
            tokens[token_key] = new_tokens[token_key]
    data["tokens"] = tokens
    data["last_refresh"] = _now_iso()
    _ensure_account_id(data)
    if auth_path is not None:
        _write_auth(auth_path, data)
    return data


def _import_codex_auth(path=None):
    source_path = Path(path) if path else _codex_cli_auth_path()
    data = _read_auth(source_path)
    tokens = dict(data.get("tokens") or {})
    if not tokens.get("access_token"):
        raise BorrowKeyError(f"No access token found in {source_path}.")
    plugin_data = _normalize_auth_data(tokens, "import")
    auth_path = _auth_path()
    _write_auth(auth_path, plugin_data)
    return auth_path, plugin_data


def _post_json(url, payload):
    status, data = _post_json_status(url, payload)
    if 200 <= status < 300:
        return data
    raise BorrowKeyError(f"Request to {url} failed (HTTP {status}): {data}")


def _post_json_status(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            body = resp.read()
            data = json.loads(body) if body else {}
            return status, data
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        try:
            error_data = json.loads(error_body)
        except Exception:
            error_data = {"error": error_body}
        return e.code, error_data
    except urllib.error.URLError as e:
        raise BorrowKeyError(f"Request to {url} failed: {e}") from None


def _pkce_pair():
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return code_verifier, code_challenge


def _exchange_authorization_code(code, code_verifier, redirect_uri=REDIRECT_URI):
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "code_verifier": code_verifier,
    }
    if redirect_uri:
        payload["redirect_uri"] = redirect_uri
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        REFRESH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise BorrowKeyError(
            f"Authorization code exchange failed (HTTP {e.code}): {error_body}"
        ) from None
    except urllib.error.URLError as e:
        raise BorrowKeyError(f"Authorization code exchange failed: {e}") from None


def _browser_login():
    code_verifier, code_challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    result = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            if query.get("state", [""])[0] != state:
                result["error"] = "OAuth state did not match."
            elif query.get("error"):
                result["error"] = query.get("error_description", query["error"])[0]
            else:
                result["code"] = query.get("code", [""])[0]
                if not result["code"]:
                    result["error"] = "OAuth callback did not include a code."

            status = 400 if result.get("error") else 200
            body = (
                "Login failed. Return to your terminal."
                if result.get("error")
                else "Login complete. Return to your terminal."
            )
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, format, *args):
            return

    try:
        server = HTTPServer(("127.0.0.1", 1455), CallbackHandler)
    except OSError as e:
        raise BorrowKeyError(f"Could not start OAuth callback server: {e}") from None
    server.timeout = 600
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "openid profile email offline_access",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    url = "https://auth.openai.com/oauth/authorize?" + urllib.parse.urlencode(params)
    click.echo("Open this URL to log in:")
    click.echo(url)
    webbrowser.open(url)
    server.handle_request()
    server.server_close()
    if result.get("error"):
        raise BorrowKeyError(result["error"])
    if not result.get("code"):
        raise BorrowKeyError("Timed out waiting for OAuth callback.")
    return _exchange_authorization_code(result["code"], code_verifier)


def _device_code_login():
    status, start = _post_json_status(DEVICE_USER_CODE_URL, {"client_id": CLIENT_ID})
    if status == 404:
        raise BorrowKeyError(
            "Device-code login is not enabled for this Codex auth server. "
            "Use `llm codex login` for browser login."
        )
    if not 200 <= status < 300:
        raise BorrowKeyError(f"Device-code request failed with status {status}: {start}")
    device_auth_id = start.get("device_auth_id")
    user_code = start.get("user_code")
    interval = int(start.get("interval") or 5)
    if not device_auth_id or not user_code:
        raise BorrowKeyError(
            "Device-code start response did not include device_auth_id and user_code."
        )
    click.echo(f"Open {DEVICE_VERIFICATION_URL}")
    click.echo(f"Enter code: {user_code}")
    started = time.monotonic()
    max_wait = 15 * 60
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= max_wait:
            raise BorrowKeyError("Device-code login timed out after 15 minutes.")
        time.sleep(min(interval, max_wait - elapsed))
        status, poll = _post_json_status(
            DEVICE_TOKEN_URL,
            {"device_auth_id": device_auth_id, "user_code": user_code},
        )
        if status in (403, 404):
            continue
        if not 200 <= status < 300:
            raise BorrowKeyError(f"Device-code login failed with status {status}: {poll}")
        error = poll.get("error") or poll.get("status")
        if error == "slow_down":
            interval += 5
            continue
        authorization_code = poll.get("authorization_code")
        code_verifier = poll.get("code_verifier")
        if authorization_code and code_verifier:
            return _exchange_authorization_code(
                authorization_code,
                code_verifier,
                redirect_uri=DEVICE_REDIRECT_URI,
            )
        if error:
            raise BorrowKeyError(f"Device-code login failed: {error}")
        raise BorrowKeyError("Device-code token response was not recognized.")


# --- Fetch available models from the Codex endpoint ---

DEFAULT_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]


def _fetch_codex_models():
    """
    Fetch the list of available models from the Codex endpoint.
    Returns a list of model slug strings. Falls back to DEFAULT_MODELS on error.
    """
    try:
        token, account_id = get_codex_key()
    except BorrowKeyError:
        return DEFAULT_MODELS

    headers = {"Authorization": f"Bearer {token}"}
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id

    req = urllib.request.Request(
        f"{CODEX_BASE_URL}/models?client_version=1.0.0",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return [
            m["slug"]
            for m in data.get("models", [])
            if m.get("supported_in_api") and m.get("visibility") == "list"
        ]
    except Exception:
        return DEFAULT_MODELS


# --- LLM Plugin ---


class ReasoningEffortEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"


class VerbosityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class CodexOptions(Options):
    model_config = ConfigDict(extra="allow")

    max_output_tokens: Optional[int] = Field(
        description="Upper bound for tokens in the response.",
        ge=0,
        default=None,
    )
    temperature: Optional[float] = Field(
        description=(
            "Sampling temperature, between 0 and 2. Higher values make output "
            "more random, lower values more focused."
        ),
        ge=0,
        le=2,
        default=None,
    )
    top_p: Optional[float] = Field(
        description="Nucleus sampling: only consider tokens in the top_p probability mass.",
        ge=0,
        le=1,
        default=None,
    )
    reasoning_effort: Optional[ReasoningEffortEnum] = Field(
        description="Reasoning effort level: low, medium, high, or xhigh.",
        default=None,
    )
    verbosity: Optional[VerbosityEnum] = Field(
        description="Controls output verbosity: low, medium, or high.",
        default=None,
    )


class _SharedCodexResponses:
    needs_key = None  # We get the key from plugin-owned Codex auth

    def __init__(self, model_name):
        self.model_id = "codex/" + model_name
        self.model_name = model_name
        self.can_stream = True
        self.supports_schema = True
        self.supports_tools = True
        self.attachment_types = {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        }
        self.Options = CodexOptions

    def __str__(self):
        return f"OpenAI Codex: {self.model_id}"

    def _get_client_kwargs(self):
        token, account_id = get_codex_key()
        headers = {}
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        return {
            "api_key": token,
            "base_url": CODEX_BASE_URL,
            "default_headers": headers,
        }

    def set_usage(self, response, usage):
        if not usage:
            return
        if not isinstance(usage, dict):
            usage = usage.model_dump()
        input_tokens = usage.pop("input_tokens")
        output_tokens = usage.pop("output_tokens")
        usage.pop("total_tokens", None)
        response.set_usage(
            input=input_tokens, output=output_tokens, details=simplify_usage_dict(usage)
        )

    def _build_messages(self, prompt, conversation):
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        attachment_message.append(
                            {"type": "input_text", "text": prev_response.prompt.prompt}
                        )
                    for attachment in prev_response.attachments:
                        attachment_message.append(_attachment(attachment))
                    messages.append({"role": "user", "content": attachment_message})
                else:
                    messages.append(
                        {"role": "user", "content": prev_response.prompt.prompt}
                    )
                for tool_result in getattr(prev_response.prompt, "tool_results", []):
                    if not tool_result.tool_call_id:
                        continue
                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_result.tool_call_id,
                            "output": tool_result.output,
                        }
                    )
                prev_text = prev_response.text_or_raise()
                if prev_text:
                    messages.append({"role": "assistant", "content": prev_text})
                tool_calls = prev_response.tool_calls_or_raise()
                if tool_calls:
                    for tool_call in tool_calls:
                        messages.append(
                            {
                                "type": "function_call",
                                "call_id": tool_call.tool_call_id,
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments),
                            }
                        )
        if not prompt.attachments:
            messages.append({"role": "user", "content": prompt.prompt or ""})
        else:
            attachment_message = []
            if prompt.prompt:
                attachment_message.append({"type": "input_text", "text": prompt.prompt})
            for attachment in prompt.attachments:
                attachment_message.append(_attachment(attachment))
            messages.append({"role": "user", "content": attachment_message})
        for tool_result in getattr(prompt, "tool_results", []):
            if not tool_result.tool_call_id:
                continue
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_result.tool_call_id,
                    "output": tool_result.output,
                }
            )
        return messages

    def _build_kwargs(self, prompt, conversation):
        messages = self._build_messages(prompt, conversation)
        kwargs = {
            "model": self.model_name,
            "input": messages,
            "store": False,
            "stream": True,
            "instructions": prompt.system or "You are a helpful assistant.",
        }
        for option in ("max_output_tokens", "temperature", "top_p"):
            value = getattr(prompt.options, option, None)
            if value is not None:
                kwargs[option] = value

        reasoning_effort = getattr(prompt.options, "reasoning_effort", None)
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        text = {}
        verbosity = getattr(prompt.options, "verbosity", None)
        if verbosity is not None:
            text["verbosity"] = verbosity

        if prompt.tools:
            tool_defs = []
            for tool in prompt.tools:
                if not getattr(tool, "name", None):
                    continue
                parameters = tool.input_schema or {
                    "type": "object",
                    "properties": {},
                }
                tool_defs.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description or None,
                        "parameters": parameters,
                        "strict": False,
                    }
                )
            if tool_defs:
                kwargs["tools"] = tool_defs
        if self.supports_schema and prompt.schema:
            text["format"] = {
                    "type": "json_schema",
                    "name": "output",
                    "schema": prompt.schema,
                }
        if text:
            kwargs["text"] = text

        extras = getattr(prompt.options, "__pydantic_extra__", None) or {}
        for key, value in extras.items():
            if value is not None and key not in kwargs:
                kwargs[key] = value
        return kwargs

    def _handle_event(self, event, response):
        et = getattr(event, "type", None)
        if et == "response.output_text.delta":
            return event.delta

        if et == "response.output_item.done":
            item = event.item
            if hasattr(item, "model_dump"):
                data = item.model_dump()
            elif isinstance(item, dict):
                data = item
            else:
                data = getattr(item, "__dict__", {}) or {}
            if data.get("type") == "function_call":
                tool_id = data.get("call_id") or data.get("id") or "unknown"
                name = data.get("name") or "unknown_tool"
                arguments = data.get("arguments") or "{}"
                try:
                    parsed = json.loads(arguments)
                except Exception:
                    parsed = arguments
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_id,
                        name=name,
                        arguments=parsed,
                    )
                )

        if et == "response.completed":
            response.response_json = event.response.model_dump()
            self.set_usage(response, event.response.usage)
            return None


class CodexResponsesModel(_SharedCodexResponses, Model):
    def execute(self, prompt, stream, response, conversation):
        client = openai.OpenAI(**self._get_client_kwargs())
        kwargs = self._build_kwargs(prompt, conversation)
        for event in client.responses.create(**kwargs):
            delta = self._handle_event(event, response)
            if delta is not None:
                yield delta


class AsyncCodexResponsesModel(_SharedCodexResponses, AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        client = openai.AsyncOpenAI(**self._get_client_kwargs())
        kwargs = self._build_kwargs(prompt, conversation)
        async for event in await client.responses.create(**kwargs):
            delta = self._handle_event(event, response)
            if delta is not None:
                yield delta


def _attachment(attachment):
    url = attachment.url
    if not url:
        base64_content = attachment.base64_content()
        url = f"data:{attachment.resolve_type()};base64,{base64_content}"
    return {"type": "input_image", "image_url": url, "detail": "low"}


def _redact(token):
    if not token:
        return ""
    if len(token) <= 14:
        return token[:4] + "..."
    return token[:8] + "..." + token[-6:]


def _exp_status(token):
    exp = _jwt_exp(token)
    if exp is None:
        return "unknown"
    iso = datetime.fromtimestamp(exp, timezone.utc).replace(microsecond=0).isoformat()
    remaining = int(exp - time.time())
    return f"{iso} ({remaining} seconds remaining)"


@click.group()
def codex():
    "Manage llm-openai-codex authentication."


@codex.command()
@click.option("--device-code", is_flag=True, help="Use device-code login.")
def login(device_code):
    "Authenticate with ChatGPT OAuth and store plugin-owned tokens."
    try:
        tokens = _device_code_login() if device_code else _browser_login()
        login_type = "chatgptDeviceCode" if device_code else "chatgpt"
        data = _normalize_auth_data(tokens, login_type)
        auth_path = _auth_path()
        _write_auth(auth_path, data)
    except BorrowKeyError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Saved llm-openai-codex auth to {auth_path}")


@codex.command()
def logout():
    "Delete plugin-owned authentication."
    auth_path = _auth_path()
    if auth_path.exists():
        auth_path.unlink()
        click.echo(f"Deleted {auth_path}")
    else:
        click.echo(f"No llm-openai-codex auth found at {auth_path}")


@codex.command()
def status():
    "Show plugin-owned authentication status."
    auth_path = _auth_path()
    try:
        data = _read_auth(auth_path)
        _ensure_account_id(data, persist_path=auth_path)
    except BorrowKeyError:
        click.echo(f"{AUTH_MISSING_MESSAGE} Auth file path: {auth_path}")
        return
    tokens = data.get("tokens") or {}
    click.echo(f"Auth file: {auth_path}")
    click.echo(f"auth_mode: {data.get('auth_mode') or ''}")
    click.echo(f"login_type: {data.get('login_type') or ''}")
    click.echo(f"account_id: {tokens.get('account_id') or ''}")
    click.echo(f"access_token: {_redact(tokens.get('access_token'))}")
    click.echo(f"access_token exp: {_exp_status(tokens.get('access_token'))}")
    if tokens.get("id_token"):
        click.echo(f"id_token exp: {_exp_status(tokens.get('id_token'))}")


@codex.command()
def refresh():
    "Refresh the stored access token."
    auth_path = _auth_path()
    try:
        data = _read_auth(auth_path)
        _refresh_auth(data, auth_path)
    except BorrowKeyError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Refreshed llm-openai-codex auth at {auth_path}")


@codex.command(name="import")
@click.option(
    "--path",
    "path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a Codex CLI auth.json file.",
)
def import_(path):
    "Import ChatGPT OAuth tokens from Codex CLI auth.json."
    try:
        auth_path, _ = _import_codex_auth(path)
    except BorrowKeyError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Imported Codex CLI auth to {auth_path}")


@hookimpl
def register_commands(cli):
    cli.add_command(codex)


@hookimpl
def register_models(register, model_aliases=None):
    model_names = _fetch_codex_models()
    for model_name in model_names:
        register(
            CodexResponsesModel(model_name),
            AsyncCodexResponsesModel(model_name),
        )
