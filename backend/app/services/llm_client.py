"""Provider-agnostic LLM access.

The agent uses an LLM for exactly two narrow jobs — rephrasing a line whose
content is already fixed, and extracting four fields from one utterance when
the rules fail. Neither needs a specific vendor, so neither should hard-code
one.

Two backends cover most of the field:

* ``anthropic`` — the Claude SDK.
* ``openai`` — the OpenAI SDK, which also speaks to anything OpenAI-compatible
  via ``LLM_BASE_URL``: OpenAI itself, Sarvam's LLM, Groq, Together, or a local
  vLLM / Ollama server. For an India-first product Sarvam is worth measuring
  here, since it is already the recommended STT/TTS vendor and is tuned on
  Indic data.

Every call is best-effort: any failure, timeout, or refusal returns ``None``
and the caller falls back to deterministic behaviour. That is what keeps the
provider choice low-stakes — a bad LLM makes the agent sound scripted, never
makes it act wrongly.
"""

from __future__ import annotations

import asyncio

from ..config import settings
from ..logger import get_logger

logger = get_logger("localmama.llm")


def available() -> bool:
    """Whether any LLM backend is configured."""
    if settings.llm_provider == "openai":
        return bool(settings.openai_api_key)
    return bool(settings.anthropic_api_key)


async def _complete_anthropic(
    *, system: str, user: str, model: str, max_tokens: int, json_schema: dict | None
) -> str | None:
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        logger.warning("anthropic package not installed")
        return None

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    output_config: dict = {}
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    # Haiku 4.5 has no extended-thinking or effort controls; passing them errors.
    if not model.startswith("claude-haiku"):
        output_config["effort"] = "low"
        kwargs["thinking"] = {"type": "disabled"}
    if output_config:
        kwargs["output_config"] = output_config

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    try:
        response = await client.messages.create(**kwargs)
    finally:
        await client.close()

    if getattr(response, "stop_reason", None) == "refusal":
        logger.info("model refused the request; falling back")
        return None
    return "".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    ).strip() or None


async def _complete_openai(
    *, system: str, user: str, model: str, max_tokens: int, json_schema: dict | None
) -> str | None:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning(
            "openai package not installed — `pip install openai` to use "
            "LLM_PROVIDER=openai"
        )
        return None

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "schema": json_schema,
                "strict": True,
            },
        }

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url or None,
    )
    try:
        response = await client.chat.completions.create(**kwargs)
    finally:
        await client.close()

    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "content_filter":
        logger.info("provider content filter tripped; falling back")
        return None
    return (choice.message.content or "").strip() or None


async def complete(
    *,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 512,
    timeout: float = 10.0,
    json_schema: dict | None = None,
) -> str | None:
    """Run one completion. Returns the text, or None on any failure.

    `json_schema` requests schema-validated JSON, which both backends support
    natively — Anthropic via `output_config.format`, OpenAI via
    `response_format.json_schema`.
    """
    if not available():
        return None

    backend = _complete_openai if settings.llm_provider == "openai" else _complete_anthropic
    try:
        return await asyncio.wait_for(
            backend(
                system=system,
                user=user,
                model=model,
                max_tokens=max_tokens,
                json_schema=json_schema,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.info("LLM call timed out after %.1fs; falling back", timeout)
        return None
    except Exception as exc:  # noqa: BLE001 - never let a provider error break a call
        logger.warning("LLM call failed (%s): %s", settings.llm_provider, exc)
        return None
