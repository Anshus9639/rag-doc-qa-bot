from typing import Iterator

import httpx
import ollama

from src.config import OLLAMA_HOST, OLLAMA_MODEL
from src.logger import get_logger

logger = get_logger(__name__)

# temperature 0 = deterministic, factual; num_ctx = how much text the model can read at once
OPTIONS = {"temperature": 0, "num_ctx": 4096}


class LLMError(Exception):
    """Raised when the local LLM can't be reached or fails."""


def _client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST, timeout=180)


def _to_llm_error(exc: Exception) -> LLMError:
    """Turn low-level errors into messages a user can act on."""
    if isinstance(exc, ollama.ResponseError) and exc.status_code == 404:
        return LLMError(
            f"Model '{OLLAMA_MODEL}' is not installed. Run:  ollama pull {OLLAMA_MODEL}"
        )
    if isinstance(exc, (ConnectionError, httpx.ConnectError)):
        return LLMError(
            f"Can't reach Ollama at {OLLAMA_HOST}. Is the Ollama app running?"
        )
    return LLMError(f"LLM request failed: {exc}")


def generate_answer(messages: list[dict]) -> str:
    """Send messages to the model and return the full answer."""
    try:
        response = _client().chat(model=OLLAMA_MODEL, messages=messages, options=OPTIONS)
    except Exception as exc:
        raise _to_llm_error(exc) from exc
    answer = (response["message"]["content"] or "").strip()
    if not answer:
        raise LLMError("The model returned an empty answer.")
    logger.info("Generated an answer of %d characters", len(answer))
    return answer


def stream_answer(messages: list[dict]) -> Iterator[str]:
    """Yield the answer piece by piece (for a typing effect in the UI)."""
    try:
        for part in _client().chat(
            model=OLLAMA_MODEL, messages=messages, options=OPTIONS, stream=True
        ):
            yield part["message"]["content"]
    except Exception as exc:
        raise _to_llm_error(exc) from exc