"""
ProcessIQ - LLM client (provider-agnostic)

Design constraints this file answers:

  1. FREE TIERS RUN OUT. This project was originally wired to Hugging Face and
     hit HTTP 402 mid-build. The client is therefore provider-agnostic: Groq,
     Hugging Face, OpenRouter and a local Ollama server all speak the same
     OpenAI-compatible chat-completions shape, so switching is configuration,
     not a rewrite.

  2. THE APP MUST SURVIVE THE LLM BEING DOWN. Every analytical tool in
     ProcessIQ is deterministic. The LLM only (a) routes a question to a tool
     and (b) writes the narrative around the numbers. If it is unavailable,
     `LLMResult.ok` is False and callers fall back to rule-based routing and
     templated narrative. The dashboard never breaks.

Configure via .env:
    LLM_PROVIDER=groq          # groq | huggingface | openrouter | ollama
    GROQ_API_KEY=gsk_...
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class Provider:
    name: str
    endpoint: str
    env_key: str
    models: tuple[str, ...]
    needs_auth: bool = True


PROVIDERS = {
    "groq": Provider(
        name="groq",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        env_key="GROQ_API_KEY",
        # Verified against GET /v1/models on this account at build time. Groq's
        # catalogue changes; run `python src/llm.py --list` to re-check.
        models=(
            "openai/gpt-oss-120b",   # strongest, best at tool routing
            "openai/gpt-oss-20b",    # fast fallback
            "qwen/qwen3.8-27b",
        ),
    ),
    "huggingface": Provider(
        name="huggingface",
        endpoint="https://router.huggingface.co/v1/chat/completions",
        env_key="HF_TOKEN",
        models=(
            "Qwen/Qwen2.5-72B-Instruct",
            "deepseek-ai/DeepSeek-V3-0324",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
    ),
    "openrouter": Provider(
        name="openrouter",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        env_key="OPENROUTER_API_KEY",
        models=(
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-2-9b-it:free",
        ),
    ),
    "ollama": Provider(
        name="ollama",
        endpoint="http://localhost:11434/v1/chat/completions",
        env_key="",
        models=("llama3.2:3b", "llama3.1:8b"),
        needs_auth=False,
    ),
}

DEFAULT_PROVIDER = (os.getenv("LLM_PROVIDER") or "groq").lower()


@dataclass
class LLMResult:
    """Outcome of a call, including which model actually served it."""
    text: str | None
    model: str | None = None
    provider: str | None = None
    error: str | None = None
    attempts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.text is not None


def _read_secret(key: str) -> str | None:
    """
    Resolve a credential from the environment, then from Streamlit secrets.

    Locally the key lives in .env. On Streamlit Community Cloud there is no
    .env file - secrets are injected through st.secrets instead - so the
    deployed app needs both paths. The import is inside the function because
    this module is also used from plain CLI scripts where streamlit may not be
    importable.
    """
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(key)
    except Exception:
        return None


class LLMClient:
    def __init__(self, provider: str | None = None):
        name = (provider or DEFAULT_PROVIDER).lower()
        if name not in PROVIDERS:
            raise ValueError(f"Unknown provider {name!r}. Options: {list(PROVIDERS)}")
        self.provider = PROVIDERS[name]
        self.api_key = _read_secret(self.provider.env_key) if self.provider.needs_auth else None

    @property
    def configured(self) -> bool:
        """True if this provider has what it needs to make a call."""
        return (not self.provider.needs_auth) or bool(self.api_key)

    def _post(self, model: str, messages: list[dict], **params) -> str:
        body = json.dumps({"model": model, "messages": messages, **params}).encode()
        # An explicit User-Agent is required: Groq sits behind Cloudflare, which
        # rejects urllib's default agent with a 403 (error code 1010).
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "processiq/1.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.provider.endpoint, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.load(resp)

        choice = payload["choices"][0]
        content = (choice["message"].get("content") or "").strip()

        # Reasoning models (Groq's gpt-oss family) emit hidden reasoning tokens
        # that are billed against max_tokens. If the budget runs out mid-thought
        # the API returns finish_reason="length" with EMPTY content. Treat that
        # as a failure so the caller retries with a larger budget rather than
        # silently rendering a blank answer.
        if not content and choice.get("finish_reason") == "length":
            raise ValueError("truncated_before_content")

        return content

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1600,
        temperature: float = 0.2,
        retries_per_model: int = 2,
    ) -> LLMResult:
        """
        Try each model in the chain; return the first success.

        The default budget is generous because reasoning models spend a large
        share of it on hidden tokens before emitting any visible content.
        """
        if not self.configured:
            return LLMResult(
                text=None,
                provider=self.provider.name,
                error=f"{self.provider.env_key} not set in .env",
                attempts=[],
            )

        attempts: list[str] = []
        last_error = None

        for model in self.provider.models:
            for attempt in range(retries_per_model):
                try:
                    text = self._post(
                        model, messages, max_tokens=max_tokens, temperature=temperature
                    )
                    attempts.append(f"{model}: ok")
                    return LLMResult(
                        text=text,
                        model=model,
                        provider=self.provider.name,
                        attempts=attempts,
                    )
                except urllib.error.HTTPError as e:
                    last_error = f"HTTP {e.code}"
                    attempts.append(f"{model}: HTTP {e.code}")
                    # 429 rate limit: short backoff then one retry on same model.
                    if e.code == 429 and attempt < retries_per_model - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
                except ValueError as e:
                    # Reasoning ran past the token budget before emitting content.
                    # Retry the same model with a doubled budget before moving on.
                    last_error = str(e)
                    attempts.append(f"{model}: {e}")
                    if attempt < retries_per_model - 1:
                        max_tokens *= 2
                        continue
                    break
                except Exception as e:
                    last_error = type(e).__name__
                    attempts.append(f"{model}: {type(e).__name__}")
                    break

        return LLMResult(
            text=None, provider=self.provider.name, error=last_error, attempts=attempts
        )

    def json_chat(self, messages: list[dict], **kwargs) -> tuple[dict | None, LLMResult]:
        """
        Chat call that must yield a JSON object.

        Models wrap JSON in prose or markdown fences even when instructed not
        to, so we extract the outermost {...} span rather than trusting the
        raw string.
        """
        result = self.chat(messages, **kwargs)
        if not result.ok:
            return None, result

        text = result.text.strip()
        if "```" in text:
            for part in text.split("```"):
                cleaned = part.removeprefix("json").strip()
                if cleaned.startswith("{"):
                    text = cleaned
                    break

        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None, result
        try:
            return json.loads(text[start : end + 1]), result
        except json.JSONDecodeError:
            return None, result


_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Module-level singleton so Streamlit reruns don't rebuild it."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def llm_available() -> bool:
    """Cheap check used by the UI to show an 'LLM offline' badge."""
    try:
        return get_client().configured
    except Exception:
        return False


if __name__ == "__main__":
    client = get_client()
    print(f"Provider : {client.provider.name}")
    print(f"Endpoint : {client.provider.endpoint}")
    print(f"Key set  : {client.configured}")

    if not client.configured:
        print(f"\nSet {client.provider.env_key} in .env to enable the narrative layer.")
        print("ProcessIQ still runs fully without it (deterministic tools).")
        raise SystemExit(0)

    print("\nTesting chain...")
    res = client.chat(
        [{"role": "user", "content": "Reply with exactly: ProcessIQ online"}], max_tokens=20
    )
    print(f"  served by: {res.model}")
    print(f"  response : {res.text!r}")
    print(f"  attempts : {res.attempts}")

    print("\nTesting JSON extraction...")
    obj, res = client.json_chat(
        [{"role": "user", "content": 'Return only JSON: {"status":"ok","n":42}'}],
        max_tokens=60,
    )
    print(f"  parsed   : {obj}")
    print(f"  attempts : {res.attempts}")
