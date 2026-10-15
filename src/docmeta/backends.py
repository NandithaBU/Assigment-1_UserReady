"""LLM backends. A backend turns (system, text, images, schema) into a dict.

Default: any OpenAI-compatible server running an open-source model
(Ollama, vLLM, LM Studio, llama.cpp server, ...). Anthropic is optional.

To support another provider, implement `Backend.extract` and pass the object
to `MetadataExtractor(backend=...)`. Nothing else in the package changes.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
from typing import Protocol, Sequence

from .models import ImagePart
from .schema import TOOL_NAME

log = logging.getLogger(__name__)

# Defaults assume a local Ollama server (https://ollama.com).
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"        # text model, used for .docx
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"        # vision model, used for scans
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # Hugging Face Hub id, runs locally
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"


class ExtractionError(RuntimeError):
    pass


class Backend(Protocol):
    def extract(
        self, system: str, text: str, images: Sequence[ImagePart], schema: dict
    ) -> dict: ...


def parse_json_object(text: str) -> dict:
    """Pull the JSON object out of a model reply.

    Open models often wrap JSON in prose or code fences, so take everything
    between the first '{' and the last '}'.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ExtractionError(f"Model reply contained no JSON object: {text[:200]!r}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Model reply was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ExtractionError("Model reply JSON was not an object.")
    return parsed


def with_json_instruction(system: str, schema: dict) -> str:
    keys = ", ".join(schema["properties"])
    return (
        f"{system}\n\nReply with a single JSON object with exactly these keys: "
        f"{keys}. Use null for any missing value. Output JSON only, no commentary."
    )


class OpenAICompatBackend:
    """Talks to any server exposing the OpenAI chat-completions API.

    Uses schema-constrained decoding when the server supports it and falls
    back to plain JSON-in-text parsing when it does not.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 vision_model: str | None = None, api_key: str | None = None,
                 max_tokens: int = 1024, timeout: float = 600.0, client=None):
        self.model = model or DEFAULT_MODEL
        self.vision_model = vision_model or (
            DEFAULT_VISION_MODEL if model is None else self.model
        )
        self.max_tokens = max_tokens
        if client is None:
            from openai import OpenAI  # lazy: only needed for real use

            client = OpenAI(
                base_url=base_url or DEFAULT_BASE_URL,
                api_key=api_key or os.getenv("DOCMETA_API_KEY", "not-needed"),
                timeout=timeout,
            )
        self._client = client

    def extract(self, system, text, images, schema):
        system_full = with_json_instruction(system, schema)
        if images:
            model = self.vision_model
            content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:{img.media_type};base64,"
                                      f"{base64.b64encode(img.data).decode('ascii')}"}}
                for img in images
            ] + [{"type": "text", "text": text}]
        else:
            model = self.model
            content = text

        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system_full},
                      {"role": "user", "content": content}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        try:
            response = self._client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": TOOL_NAME, "schema": schema}},
            )
        except Exception as exc:  # server without schema-constrained output
            log.debug("json_schema response_format failed (%s); retrying without it", exc)
            response = self._client.chat.completions.create(**kwargs)
        return parse_json_object(response.choices[0].message.content or "")


class TransformersBackend:
    """Runs a Hugging Face model in-process (`pip install 'docmeta[hf]'`).

    Text only: scans must go through OCR (`--ocr`). The model is downloaded
    from the Hugging Face Hub on first use and cached. Plain transformers has
    no schema-constrained decoding, so the JSON reply is parsed tolerantly.
    """

    def __init__(self, model: str | None = None, device_map: str = "auto",
                 max_new_tokens: int = 400, pipeline=None):
        self.model = model or DEFAULT_HF_MODEL
        self.max_new_tokens = max_new_tokens
        if pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline  # lazy: heavy import

            if device_map == "auto" and not torch.cuda.is_available():
                # On a CPU-only machine "auto" can spill weights to disk, which is
                # extremely slow. Keep everything in RAM instead.
                device_map = "cpu"
            log.info("Loading %s on %s (first call is slow)", self.model, device_map)
            pipeline = hf_pipeline("text-generation", model=self.model,
                                   dtype="auto", device_map=device_map)
        self._pipe = pipeline
        self._lock = threading.Lock()  # one generation at a time per loaded model

    def extract(self, system, text, images, schema):
        if images:
            raise ExtractionError(
                "The transformers provider is text-only. Use --ocr for scanned images, "
                "or serve a vision model with vLLM/Ollama and use --provider openai."
            )
        messages = [{"role": "system", "content": with_json_instruction(system, schema)},
                    {"role": "user", "content": text}]
        with self._lock:
            out = self._pipe(messages, max_new_tokens=self.max_new_tokens,
                             do_sample=False, return_full_text=False)
        generated = out[0]["generated_text"]
        if isinstance(generated, list):  # chat-style output: list of messages
            generated = generated[-1]["content"]
        return parse_json_object(generated)


class AnthropicBackend:
    """Optional: Claude with forced tool use (`pip install docmeta[anthropic]`)."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_tokens: int = 1024):
        import anthropic

        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def extract(self, system, text, images, schema):
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": img.media_type,
                                         "data": base64.b64encode(img.data).decode("ascii")}}
            for img in images
        ]
        content.append({"type": "text", "text": text})
        response = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            tools=[{"name": TOOL_NAME,
                    "description": "Record the metadata extracted from the agreement.",
                    "input_schema": schema}],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ExtractionError("Model did not return structured output.")
