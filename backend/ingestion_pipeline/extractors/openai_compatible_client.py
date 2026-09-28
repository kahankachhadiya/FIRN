"""OpenAI-compatible client for document processor vision and text tasks.

Replaces the old LMStudioClient with a provider-agnostic implementation that
works with local (LM Studio, Ollama) and cloud (NVIDIA NIM, OpenAI, Groq, etc.)
endpoints.

Key behaviours:
- Bearer token included only when api_key is non-empty (cloud providers require it;
  local providers typically do not).
- Response content extracted from `message.content` with fallback to
  `message.reasoning` for thinking/reasoning models (e.g. Kimi K2.5, DeepSeek R1).
- Hard errors on bad responses — no silent empty-string returns.
- Retry with exponential backoff on 5xx / connection errors.
"""

import json
import base64
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


# ── Public exceptions ─────────────────────────────────────────────────────────

class OpenAICompatibleError(Exception):
    """Raised when the API returns an error or all retries are exhausted."""
    pass


# Keep the old name as an alias so existing catch-sites still work.
LMStudioError = OpenAICompatibleError


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class LMStudioResponse:
    """Parsed response from an OpenAI-compatible endpoint."""
    content: str
    usage: Dict[str, int]
    model: str


# ── Client ────────────────────────────────────────────────────────────────────

class OpenAICompatibleClient:
    """Provider-agnostic OpenAI-compatible HTTP client.

    Works with:
    - Local providers  : LM Studio, Ollama, vLLM  (api_key="" → no Auth header)
    - Cloud providers  : NVIDIA NIM, OpenAI, Groq, Anthropic-compat endpoints
                         (api_key non-empty → Authorization: Bearer <key>)

    Handles reasoning models (Kimi K2.5, DeepSeek R1, QwQ) that return the
    generated text in `message.reasoning` instead of `message.content`.
    """

    def __init__(
        self,
        image_endpoint: str,
        api_key: str,
        image_model: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            image_endpoint: Base URL of the OpenAI-compatible server,
                            e.g. "https://integrate.api.nvidia.com/v1"
                            or "http://localhost:1234/v1".
                            Must NOT end with /chat/completions — the client
                            appends the path itself.
            api_key:        Bearer token.  Pass "" for local providers.
            image_model:    Model identifier, e.g. "moonshotai/kimi-k2.5".
            max_tokens:     Default max_tokens for completions.
            temperature:    Default sampling temperature.
            max_retries:    Retry attempts on 5xx / connection errors.
            retry_delay:    Base delay (seconds) for exponential backoff.
        """
        if not image_endpoint:
            raise ValueError("image_endpoint must not be empty")
        if not image_model:
            raise ValueError("image_model must not be empty")

        self.api_key = api_key
        self.image_model = image_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Normalise URL: accept full completions URL or base URL
        raw = image_endpoint.rstrip("/")
        if raw.endswith("/chat/completions"):
            self.image_chat_url = raw
            self.image_endpoint = raw[: -len("/chat/completions")]
        elif raw.endswith("/v1"):
            self.image_endpoint = raw
            self.image_chat_url = raw + "/chat/completions"
        else:
            self.image_endpoint = raw
            self.image_chat_url = raw + "/v1/chat/completions"

        print(f"OpenAICompatibleClient: endpoint  = {self.image_endpoint}")
        print(f"OpenAICompatibleClient: model     = {self.image_model}")
        print(f"OpenAICompatibleClient: max_tokens= {self.max_tokens}")
        print(f"OpenAICompatibleClient: auth      = {'Bearer ***' if self.api_key else 'none (local)'}")

    # ── Headers ───────────────────────────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        """Return request headers.  Bearer token only when api_key is set."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── Content extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_content(message: Dict) -> str:
        """Extract text from a chat completion message dict.

        Standard models put text in `content`.
        Reasoning/thinking models (Kimi K2.5, DeepSeek R1, QwQ, o1) put the
        visible answer in `reasoning` when `content` is null.
        """
        return message.get("content") or message.get("reasoning") or ""

    # ── HTTP with retry ───────────────────────────────────────────────────────

    def _post_with_retry(self, url: str, payload: Dict) -> requests.Response:
        """POST *payload* to *url* with retry on 5xx/network errors and 429 rate-limits.

        429 (Too Many Requests) uses a fixed back-off schedule:
            attempt 1 failed → wait 2 s
            attempt 2 failed → wait 30 s
            attempt 3 failed → raise

        5xx / network errors use exponential back-off as before.

        Raises:
            OpenAICompatibleError: after all retries are exhausted.
        """
        # Delays applied AFTER each failed attempt (index = attempt number, 0-based)
        RATE_LIMIT_DELAYS = [2, 30]   # wait 2 s after 1st fail, 30 s after 2nd fail
        MAX_ATTEMPTS = 3              # 1 original + 2 retries

        headers = self._build_headers()
        last_exc: Optional[Exception] = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=None)

                # ── 429 Rate-limit ────────────────────────────────────────────
                if response.status_code == 429:
                    if attempt < MAX_ATTEMPTS - 1:
                        delay = RATE_LIMIT_DELAYS[attempt]
                        print(
                            f"OpenAICompatibleClient: rate-limited (429) "
                            f"(attempt {attempt + 1}/{MAX_ATTEMPTS}) — "
                            f"retrying in {delay}s…"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        # All retries exhausted on 429 — let caller decide what to do
                        raise OpenAICompatibleError(
                            f"Rate-limited (429) after {MAX_ATTEMPTS} attempts — skipping image"
                        )

                # ── 5xx server errors — exponential back-off ──────────────────
                if response.status_code >= 500:
                    if attempt < MAX_ATTEMPTS - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        print(
                            f"OpenAICompatibleClient: server error {response.status_code} "
                            f"(attempt {attempt + 1}/{MAX_ATTEMPTS}) — retrying in {delay:.1f}s…"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise OpenAICompatibleError(
                            f"Server error {response.status_code} after {MAX_ATTEMPTS} attempts"
                        )

                # ── 2xx / 4xx (not 429) — return immediately ──────────────────
                return response

            except OpenAICompatibleError:
                raise  # propagate our own errors without wrapping

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as exc:
                last_exc = exc
                if attempt < MAX_ATTEMPTS - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(
                        f"OpenAICompatibleClient: network error "
                        f"(attempt {attempt + 1}/{MAX_ATTEMPTS}): {exc} — retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                else:
                    raise OpenAICompatibleError(
                        f"Network error after {MAX_ATTEMPTS} attempts. Last error: {last_exc}"
                    )

    def _get_with_retry(self, url: str) -> requests.Response:
        """GET *url* with retry logic."""
        headers = self._build_headers()
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code < 500:
                    return response
                print(
                    f"OpenAICompatibleClient: server error {response.status_code} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                print(f"OpenAICompatibleClient: GET error (attempt {attempt + 1}): {exc}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))

        raise OpenAICompatibleError(
            f"GET failed after {self.max_retries} attempts. Last error: {last_exc}"
        )

    # ── Core completion ───────────────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = True,
    ) -> str:
        """Send a text chat completion request.

        Args:
            messages:    OpenAI-format message list.
            max_tokens:  Override instance default.
            temperature: Override instance default.
            json_mode:   Unused (kept for API compatibility); prompts should
                         request JSON explicitly.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            OpenAICompatibleError: On HTTP error or exhausted retries.
        """
        payload = {
            "model": self.image_model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }

        response = self._post_with_retry(self.image_chat_url, payload)

        if response.status_code != 200:
            raise OpenAICompatibleError(
                f"chat_completion failed: HTTP {response.status_code} — {response.text[:400]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise OpenAICompatibleError("chat_completion: API returned no choices")

        content = self._extract_content(choices[0].get("message", {}))
        if not content:
            raise OpenAICompatibleError(
                "chat_completion: API returned empty content and empty reasoning field. "
                f"Full response: {json.dumps(data)[:600]}"
            )
        return content

    # ── Vision completion ─────────────────────────────────────────────────────

    def vision_completion(
        self,
        image_path: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send an image + text prompt to the vision endpoint.

        Args:
            image_path: Local path to the image file.
            prompt:     Text instruction for the model.
            max_tokens: Override instance default.
            temperature: Override instance default.

        Returns:
            The assistant's reply as a plain string (may be JSON).

        Raises:
            OpenAICompatibleError: On encoding failure, HTTP error, or empty response.
        """
        image_data = self._encode_image(image_path)

        ext = Path(image_path).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/jpeg")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                ],
            }
        ]

        payload = {
            "model": self.image_model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }

        response = self._post_with_retry(self.image_chat_url, payload)

        if response.status_code != 200:
            raise OpenAICompatibleError(
                f"vision_completion failed: HTTP {response.status_code} — {response.text[:400]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise OpenAICompatibleError("vision_completion: API returned no choices")

        content = self._extract_content(choices[0].get("message", {}))
        if not content:
            raise OpenAICompatibleError(
                "vision_completion: API returned empty content and empty reasoning field. "
                f"Full response: {json.dumps(data)[:600]}"
            )
        return content

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_image(self, image_path: str) -> str:
        """Base64-encode an image file.

        Raises:
            OpenAICompatibleError: If the file cannot be read.
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except OSError as exc:
            raise OpenAICompatibleError(f"Cannot read image file {image_path!r}: {exc}") from exc

    # ── High-level pipeline helpers ───────────────────────────────────────────

    def chunk_markdown(self, markdown: str, prompt: str) -> List[Dict]:
        """Chunk markdown using the LLM.  Returns list of chunk dicts."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": markdown},
        ]
        raw = self.chat_completion(messages, max_tokens=4000)
        try:
            return json.loads(raw).get("chunks", [])
        except json.JSONDecodeError:
            print(f"OpenAICompatibleClient: chunk_markdown — could not parse JSON response")
            return []

    def extract_metadata(self, markdown: str, prompt: str) -> Dict:
        """Extract document metadata from the first 3000 chars of markdown."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": markdown[:3000]},
        ]
        raw = self.chat_completion(messages, max_tokens=1000)
        try:
            return json.loads(raw).get("document_metadata", {})
        except json.JSONDecodeError:
            print("OpenAICompatibleClient: extract_metadata — could not parse JSON response")
            return {}

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Analyse an image and return the analysis text."""
        raw = self.vision_completion(image_path, prompt)
        try:
            return json.loads(raw).get("image_analysis_text", raw)
        except json.JSONDecodeError:
            return raw  # plain-text response from some models

    def test_connection(self) -> bool:
        """Probe the /v1/models endpoint to verify connectivity.

        Returns True on success, raises OpenAICompatibleError on failure.
        """
        print("OpenAICompatibleClient: testing connection…")
        try:
            base = self.image_endpoint
            models_url = (
                f"{base}/models" if base.endswith("/v1")
                else f"{base}/v1/models"
            )
            response = self._get_with_retry(models_url)
            if response.status_code == 200:
                models = [m["id"] for m in response.json().get("data", [])]
                print(f"✓ Connection OK — available models: {models or '(none listed)'}")
                return True
            # Some cloud providers (NVIDIA NIM) don't expose /v1/models but still work.
            # Treat non-200 as a warning, not a hard failure.
            print(
                f"⚠ /v1/models returned {response.status_code} — "
                "endpoint may still be functional (some providers omit this route)"
            )
            return True
        except OpenAICompatibleError as exc:
            print(f"✗ Connection test failed: {exc}")
            raise


# ── Backward-compatibility alias ──────────────────────────────────────────────
# Any code that still imports LMStudioClient will get the new implementation.
LMStudioClient = OpenAICompatibleClient
