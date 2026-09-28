"""
Gating Router — Hybrid Cascading Classifier

Two-stage classification strategy:
  - Fast Path: ClassificationServiceClient (ModernBERT-large-zeroshot)
  - Slow Path: async LLM fallback via LM Studio for low-confidence results

Behaviour controlled by CONFIDENCE_THRESHOLD env var:
  - CONFIDENCE_THRESHOLD=0   → Slow Path disabled; all Fast Path results returned as-is
  - CONFIDENCE_THRESHOLD>0   → Slow Path enabled; results below threshold sent to LLM

The GatingRouter is a transparent drop-in for ClassificationServiceClient.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml

from external_clients.classification_client import (
    ClassificationResponse,
    ClassificationResult,
    ClassificationServiceClient,
)
from external_clients.openai_compatible_client import OpenAICompatibleClient, OpenAICompatibleError
from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_classification_prompt() -> str:
    """Load the slow-path classification prompt from config/prompts.yml.

    Raises:
        FileNotFoundError: If prompts.yml does not exist.
        KeyError: If 'classification_slow_path_prompt' key is missing.
    """
    prompts_path = Path(__file__).parent.parent / "config" / "prompts.yml"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts configuration file not found: {prompts_path}. "
            "Ensure config/prompts.yml exists in the ingestion_pipeline directory."
        )
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "classification_slow_path_prompt" not in data:
        raise KeyError(
            "prompts.yml is missing required key 'classification_slow_path_prompt'. "
            "Add it to config/prompts.yml."
        )
    return data["classification_slow_path_prompt"].strip()


_CLASSIFICATION_PROMPT_TEMPLATE: str = _load_classification_prompt()

# ---------------------------------------------------------------------------

# Read threshold from env — must be set explicitly (no hardcoded default)
_raw = os.getenv("CONFIDENCE_THRESHOLD")
if _raw is None:
    raise EnvironmentError(
        "CONFIDENCE_THRESHOLD environment variable is not set. "
        "Add it to your .env.production file. "
        "Set to 0 to disable the Slow Path, or a value in (0, 1] to enable it."
    )
CONFIDENCE_THRESHOLD: float = float(_raw)


class GatingRouter:
    """
    Hybrid cascading classifier that wraps ClassificationServiceClient.

    When CONFIDENCE_THRESHOLD > 0:
      - Results >= threshold pass through (Fast Path only)
      - Results < threshold are re-evaluated by the LLM (Slow Path)

    When CONFIDENCE_THRESHOLD == 0:
      - Slow Path is disabled entirely
      - All Fast Path results are returned as-is
    """

    VALID_RELATIONS = {"dependency", "expansion", "contradiction", "unrelated"}

    def __init__(
        self,
        classification_client: Optional[ClassificationServiceClient] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {threshold}"
            )

        self.confidence_threshold: float = threshold
        self.classification_client = classification_client or ClassificationServiceClient()

        # Slow Path is enabled only when the flag is true AND threshold > 0
        self._slow_path_enabled: bool = (
            threshold > 0.0 and settings.classification_slow_path_enabled
        )

        if self._slow_path_enabled:
            if not settings.llm_classification_base_url:
                logger.warning(
                    "LLM_CLASSIFICATION_BASE_URL is empty — Slow Path will be skipped despite threshold=%.2f.",
                    threshold,
                )
                self._slow_path_enabled = False
                self._llm_client: Optional[OpenAICompatibleClient] = None
            else:
                self._llm_client = OpenAICompatibleClient(
                    image_endpoint=settings.llm_classification_base_url,
                    api_key=settings.llm_classification_api_key,
                    image_model=settings.llm_classification_model,
                    max_tokens=settings.llm_classification_max_tokens,
                    temperature=settings.llm_classification_temperature,
                    max_retries=3,
                    retry_delay=1.0,
                )
        else:
            if not settings.classification_slow_path_enabled:
                logger.info("GatingRouter: CLASSIFICATION_SLOW_PATH_ENABLED=false — Slow Path disabled.")
            else:
                logger.info("GatingRouter: CONFIDENCE_THRESHOLD=0 — Slow Path disabled.")

        logger.info(
            "GatingRouter initialized: confidence_threshold=%.2f, slow_path=%s",
            self.confidence_threshold,
            self._slow_path_enabled,
        )

    def _get_slow_path_client(self) -> "OpenAICompatibleClient":
        """Return the OpenAICompatibleClient for the Slow Path."""
        return self._llm_client  # type: ignore[return-value]

    async def close(self) -> None:
        """Gracefully close all internal async clients."""
        if (
            self.classification_client._async_client
            and not self.classification_client._async_client.is_closed
        ):
            await self.classification_client._async_client.aclose()

    # ------------------------------------------------------------------
    # Public interface (drop-in for ClassificationServiceClient)
    # ------------------------------------------------------------------

    def classify_relationships(
        self,
        text_pairs: List[Dict[str, str]],
        batch_size: Optional[int] = None,
    ) -> ClassificationResponse:
        """
        Synchronous entry point — drop-in for ClassificationServiceClient.

        Handles both cases:
        - No running event loop: uses asyncio.run()
        - Already inside a running loop: offloads to a thread pool
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(self._classify_async(text_pairs, batch_size))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, self._classify_async(text_pairs, batch_size)
                )
                return future.result()

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _classify_async(
        self,
        text_pairs: List[Dict[str, str]],
        batch_size: Optional[int] = None,
    ) -> ClassificationResponse:
        """Two-stage classification pipeline."""
        start_time = time.monotonic()

        # Reset client to avoid closed-loop issues when called from a thread pool
        self.classification_client._async_client = None
        fast_response: ClassificationResponse = (
            await self.classification_client.classify_relationships_async(text_pairs, batch_size)
        )

        # If Slow Path is disabled, return only results at or above the threshold
        if not self._slow_path_enabled:
            kept = [r for r in fast_response.results if r.confidence_score >= self.confidence_threshold]
            dropped = len(fast_response.results) - len(kept)
            if dropped:
                logger.info("Slow Path disabled: dropped %d/%d pairs below threshold %.2f",
                            dropped, len(fast_response.results), self.confidence_threshold)
            processing_time = time.monotonic() - start_time
            return ClassificationResponse(
                results=kept,
                processing_time=processing_time,
                model_info=fast_response.model_info,
            )

        # Triage: split into confident and weak
        confident_results, weak_results = self._triage(fast_response.results)

        if not weak_results:
            processing_time = time.monotonic() - start_time
            return ClassificationResponse(
                results=sorted(confident_results, key=lambda r: r.id),
                processing_time=processing_time,
                model_info=fast_response.model_info,
            )

        # Slow Path — fire all weak results concurrently
        client = self._get_slow_path_client()
        tasks = [self._slow_path_single(result, client) for result in weak_results]
        resolved_weak: List[ClassificationResult] = list(await asyncio.gather(*tasks))

        processing_time = time.monotonic() - start_time
        merged = sorted(confident_results + resolved_weak, key=lambda r: r.id)
        return ClassificationResponse(
            results=merged,
            processing_time=processing_time,
            model_info=fast_response.model_info,
        )

    # ------------------------------------------------------------------
    # Triage
    # ------------------------------------------------------------------

    def _triage(
        self,
        results: List[ClassificationResult],
    ) -> tuple[List[ClassificationResult], List[ClassificationResult]]:
        """Split results into (confident, weak) based on confidence_threshold."""
        confident: List[ClassificationResult] = []
        weak: List[ClassificationResult] = []
        for result in results:
            if result.confidence_score >= self.confidence_threshold:
                confident.append(result)
            else:
                weak.append(result)
        return confident, weak

    # ------------------------------------------------------------------
    # Slow Path — prompt + request + parsing
    # ------------------------------------------------------------------

    def _build_prompt(self, result: ClassificationResult) -> str:
        """Build the Chain-of-Thought prompt for a single weak result."""
        text_a = result.text_a[:400]
        text_b = result.text_b[:400]
        return _CLASSIFICATION_PROMPT_TEMPLATE.format(text_a=text_a, text_b=text_b)

    async def _slow_path_single(
        self,
        result: ClassificationResult,
        client: "OpenAICompatibleClient",
    ) -> ClassificationResult:
        """Send one weak result to the LLM via OpenAICompatibleClient; fall back to Fast Path result on any failure."""
        messages = [{"role": "user", "content": self._build_prompt(result)}]
        try:
            body = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat_completion(messages),
            )
            if not body:
                logger.warning("Slow Path empty content for id=%s", result.id)
                return result
            return self._parse_llm_response(body, result)

        except OpenAICompatibleError as exc:
            logger.warning("Slow Path error for id=%s: %s — retaining Fast Path result.", result.id, exc)
            return result

        except Exception as exc:
            logger.warning("Slow Path error for id=%s: %s — retaining Fast Path result.", result.id, exc)
            return result

    def _parse_llm_response(
        self,
        body: str,
        original: ClassificationResult,
    ) -> ClassificationResult:
        """Parse LLM response; return original on any failure."""
        try:
            # Strip <think>...</think> blocks (Qwen3)
            body = re.sub(r"<think>.*?</think>", "", body, flags=re.DOTALL).strip()
            # Strip markdown code fences
            body = re.sub(r"^```(?:json)?\s*", "", body).strip()
            body = re.sub(r"\s*```$", "", body).strip()

            if not body:
                logger.warning("LLM empty body for id=%s — retaining Fast Path result.", original.id)
                return original

            data = json.loads(body)
            relation = data.get("relation", "")
            if relation not in self.VALID_RELATIONS:
                logger.warning("LLM invalid relation %r for id=%s — retaining Fast Path result.",
                               relation, original.id)
                return original

            return ClassificationResult(
                id=original.id,
                relation_type=relation,
                confidence_score=0.99,
                text_a=original.text_a,
                text_b=original.text_b,
                debug_scores=original.debug_scores,
                metadata=original.metadata,
            )

        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning("Failed to parse LLM response for id=%s: %s — retaining Fast Path result.",
                           original.id, exc)
            return original
