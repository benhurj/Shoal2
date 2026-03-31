# llm_client.py
import asyncio
import base64
import re
import math
import httpx
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

_RETRY_DELAYS = [2, 5, 10]       # seconds between retries on 429
_COLD_START_DELAYS = [10, 20, 30, 60, 90]  # seconds between retries on 303 (cold start)
from env_config import get_llm_config
from config import LLM_MAX_TOKENS, MAX_CONCURRENT_LLM_CALLS, LOGPROBS_TOP_K

logger = structlog.get_logger()


@dataclass
class TokenDistribution:
    """Represents the probability distribution for a single token position."""
    token: str
    logprob: float
    probability: float
    token_id: Optional[int] = None
    top_alternatives: Optional[List[Tuple[str, float]]] = None  # (token, prob) pairs


@dataclass
class LLMResponse:
    """Extended LLM response with distribution information."""
    content: str
    total_tokens: int
    token_distributions: Optional[List[Dict[str, float]]] = None  # {token: prob} for each position
    token_logprobs: Optional[List[float]] = None
    tokens: Optional[List[str]] = None
    logits: Optional[Any] = None  # numpy array (num_tokens, vocab_size) — raw pre-softmax logits
    vocab_size: Optional[int] = None


@dataclass
class BranchResult:
    """Result from a single branch of branched generation."""
    content: str
    tokens_used: int
    temperature: float
    top_p: float


@dataclass
class BranchedLLMResponse:
    """Response from branched generation — K outputs from one forward-pass loop."""
    branches: List[BranchResult]
    prompt_tokens: int
    vocab_size: int


class LLMClient:
    """Unified LLM client supporting local Ollama, Ollama Cloud, and OpenAI"""

    def __init__(self):
        self.config = get_llm_config()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
        self._async_client: httpx.AsyncClient | None = None

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Return a shared async client, creating it on first use."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=600.0,
                follow_redirects=True,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._async_client

    def _build_openai_payload(self, messages: List[Dict], model: str, **overrides) -> Dict:
        """Build payload for OpenAI-compatible API"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", LLM_MAX_TOKENS),
            "temperature": overrides.get("temperature", 0.5),
        }
        if "top_p" in overrides:
            payload["top_p"] = overrides["top_p"]
        if "stop" in overrides:
            payload["stop"] = overrides["stop"]
        return payload

    def _build_native_ollama_payload(self, messages: List[Dict], model: str, **overrides) -> Dict:
        """Build payload for Ollama native /api/chat endpoint."""
        options = {
            "num_predict": overrides.get("max_tokens", LLM_MAX_TOKENS),
            "temperature": overrides.get("temperature", 0.5),
        }
        if "top_p" in overrides:
            options["top_p"] = overrides["top_p"]
        if "stop" in overrides:
            options["stop"] = overrides["stop"]
        return {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }

    async def call_llm(
        self,
        messages: List[Dict],
        model: str,
        stop: Optional[List[str]] = None,
        **overrides
    ) -> Tuple[str, int]:
        """Unified LLM call that routes to appropriate service"""
        worker_idx = overrides.pop("worker_idx", None)
        async with self._semaphore:
            if self.config["type"] == "modal":
                return await self._call_modal(messages, model, stop, worker_idx=worker_idx, **overrides)
            elif self.config["type"] == "hybrid":
                from config import WORKER_MODEL
                if model == WORKER_MODEL:
                    return await self._call_local_openai_compat(messages, model, stop, **overrides)
                else:
                    return await self._call_modal(messages, model, stop, **overrides)
            elif self.config["type"] == "ollama_cloud":
                return await self._call_ollama_cloud(messages, model, stop, **overrides)
            elif self.config["type"] == "openai":
                return await self._call_openai(messages, model, stop, **overrides)
            else:
                return await self._call_local_ollama(messages, model, stop, **overrides)

    async def _call_ollama_cloud(
        self,
        messages: List[Dict],
        model: str,
        stop: Optional[List[str]],
        **overrides
    ) -> Tuple[str, int]:
        """Call Ollama Cloud API"""
        headers = {"Authorization": f"Bearer {self.config['api_key']}"}

        if stop:
            overrides['stop'] = stop

        payload = self._build_openai_payload(messages, model, **overrides)
        client = await self._get_async_client()

        last_err = None
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            try:
                resp = await client.post(
                    f"{self.config['base_url']}/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
                resp.raise_for_status()
                return self._parse_openai_response(resp.json())
            except httpx.TimeoutException:
                raise RuntimeError(f"LLM call timed out (Ollama Cloud, model {model})")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    last_err = e
                    logger.warning("ollama_cloud_429_retry", attempt=attempt + 1, delay=_RETRY_DELAYS[attempt] if attempt < len(_RETRY_DELAYS) else "giving up")
                    continue
                raise RuntimeError(f"LLM returned HTTP {e.response.status_code} (Ollama Cloud)")
        raise RuntimeError(f"LLM returned HTTP 429 (Ollama Cloud) after {len(_RETRY_DELAYS) + 1} attempts")

    async def _call_openai(
        self,
        messages: List[Dict],
        model: str,
        stop: Optional[List[str]],
        **overrides
    ) -> Tuple[str, int]:
        """Call OpenAI API"""
        headers = {"Authorization": f"Bearer {self.config['api_key']}"}

        if stop:
            overrides['stop'] = stop

        payload = self._build_openai_payload(messages, model, **overrides)
        client = await self._get_async_client()

        try:
            resp = await client.post(
                f"{self.config['base_url']}/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(f"LLM call timed out (OpenAI, model {model})")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM returned HTTP {e.response.status_code} (OpenAI)")

        return self._parse_openai_response(resp.json())

    async def _call_local_ollama(
        self,
        messages: List[Dict],
        model: str,
        stop: Optional[List[str]],
        **overrides
    ) -> Tuple[str, int]:
        """Call local Ollama instance using native /api/chat (supports think control)."""
        if stop:
            overrides['stop'] = stop

        payload = self._build_native_ollama_payload(messages, model, **overrides)

        port = overrides.get("port", 11434)
        url = f"{self.config['base_url']}:{port}/api/chat"

        loop = asyncio.get_event_loop()

        def make_request():
            with httpx.Client(timeout=600.0) as client:
                try:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json()
                except httpx.TimeoutException:
                    raise RuntimeError(f"LLM call timed out (port {port}, model {model})")
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(f"LLM returned HTTP {e.response.status_code} (port {port})")

        try:
            data = await loop.run_in_executor(self.executor, make_request)
            return self._parse_native_response(data)
        except Exception as e:
            logger.error("local_ollama_failed", error=str(e), port=port, model=model)
            raise

    # ── Local OpenAI-compatible server (Ollama /v1) ──────────────────────

    async def _call_local_openai_compat(
        self,
        messages: List[Dict],
        model: str,
        stop: Optional[List[str]],
        **overrides,
    ) -> Tuple[str, int]:
        """Call a local Ollama server at LOCAL_WORKER_URL (e.g. http://localhost:11434/v1)."""
        if stop:
            overrides["stop"] = stop
        payload = self._build_openai_payload(messages, model, **overrides)
        url = f"{self.config['worker_url'].rstrip('/')}/chat/completions"
        api_key = self.config.get("worker_api_key", "none")
        headers = {"Authorization": f"Bearer {api_key}"}
        client = await self._get_async_client()
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(f"Local worker timed out (url={url}, model={model})")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Local worker returned HTTP {e.response.status_code} (url={url})")
        return self._parse_openai_response(resp.json())

    # ── Modal (full logits) ─────────────────────────────────────────────

    async def _modal_post_with_retry(self, url: str, payload: dict) -> dict:
        """POST to Modal with retry on 303 (cold start) and 429 (rate limit)."""
        client = await self._get_async_client()
        last_err = None
        for attempt, delay in enumerate([0] + _COLD_START_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException:
                raise RuntimeError(f"Modal call timed out after {attempt + 1} attempts")
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code in (303, 429, 502, 503):
                    logger.warning("modal_retry", status=e.response.status_code,
                                 attempt=attempt + 1, url=url)
                    continue
                body = e.response.text[:500] if e.response.text else ""
                logger.error("modal_http_error", status=e.response.status_code, body=body)
                raise RuntimeError(f"Modal returned HTTP {e.response.status_code}")
        status = last_err.response.status_code if last_err else "unknown"
        raise RuntimeError(f"Modal call failed after {len(_COLD_START_DELAYS) + 1} retries (last: {status})")

    async def _call_modal(
        self,
        messages: List[Dict],
        model: str,
        stop: Optional[List[str]],
        worker_idx: Optional[int] = None,
        **overrides
    ) -> Tuple[str, int]:
        """Call Modal endpoint (text generation, no logits)."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", LLM_MAX_TOKENS),
            "temperature": overrides.get("temperature", 0.5),
            "return_logits": False,
        }
        if "top_p" in overrides:
            payload["top_p"] = overrides["top_p"]
        if stop:
            payload["stop"] = stop
        if worker_idx is not None:
            payload["worker_idx"] = worker_idx

        base = self._modal_url_for(model)
        url = f"{base}/generate"

        data = await self._modal_post_with_retry(url, payload)
        content = data.get("content", "")
        # Strip Qwen3 thinking blocks (inline or tagged)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content:
            content = data.get("content", "").strip()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return content, tokens

    def _modal_url_for(self, model: str) -> str:
        """Return the Modal endpoint URL (single container serves all models)."""
        url = self.config.get("modal_url") or self.config.get("base_url", "")
        return url.rstrip("/")

    async def call_llm_with_logits(
        self,
        messages: List[Dict],
        model: str,
        stop: Optional[List[str]] = None,
        **overrides
    ) -> LLMResponse:
        """Generate text and return full raw logits (Modal only).

        Returns an LLMResponse with .logits as a numpy array of shape
        (num_generated_tokens, vocab_size) containing pre-softmax logits.
        """
        if self.config["type"] != "modal":
            raise RuntimeError(
                "call_llm_with_logits requires Modal deployment mode "
                f"(current: {self.config['type']})"
            )

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", LLM_MAX_TOKENS),
            "temperature": overrides.get("temperature", 0.5),
            "return_logits": True,
        }
        if "top_p" in overrides:
            payload["top_p"] = overrides["top_p"]
        if stop:
            payload["stop"] = stop

        async with self._semaphore:
            base = self._modal_url_for(model)
            url = f"{base}/generate"
            data = await self._modal_post_with_retry(url, payload)

        logits = None
        vocab_size = data.get("vocab_size")
        if "logits_b64" in data:
            logits = np.frombuffer(
                base64.b64decode(data["logits_b64"]), dtype=np.float32
            ).reshape(data["logits_shape"])

        return LLMResponse(
            content=data.get("content", ""),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
            tokens=data.get("tokens"),
            logits=logits,
            vocab_size=vocab_size,
        )

    async def call_llm_branched(
        self,
        messages: List[Dict],
        model: str,
        branch_configs: List[Dict],
        stop: Optional[List[str]] = None,
        worker_idx: Optional[int] = None,
        **overrides,
    ) -> BranchedLLMResponse:
        """Generate K branched outputs from a single autoregressive loop (Modal only).

        Each branch_config is a dict with 'temperature' and 'top_p'.
        All branches share the same prompt and KV cache prefix.
        """
        if self.config["type"] != "modal":
            raise RuntimeError(
                "call_llm_branched requires Modal deployment mode "
                f"(current: {self.config['type']})"
            )

        payload = {
            "model": model,
            "messages": messages,
            "branch_configs": branch_configs,
            "max_tokens": overrides.get("max_tokens", LLM_MAX_TOKENS),
        }
        if stop:
            payload["stop"] = stop
        if worker_idx is not None:
            payload["worker_idx"] = worker_idx

        async with self._semaphore:
            base = self._modal_url_for(model)
            url = f"{base}/generate_branched"
            data = await self._modal_post_with_retry(url, payload)

        branches = []
        for i, b in enumerate(data.get("branches", [])):
            cfg = branch_configs[i] if i < len(branch_configs) else {}
            branches.append(BranchResult(
                content=b.get("content", ""),
                tokens_used=b.get("tokens_used", 0),
                temperature=cfg.get("temperature", 0.5),
                top_p=cfg.get("top_p", 1.0),
            ))

        return BranchedLLMResponse(
            branches=branches,
            prompt_tokens=data.get("prompt_tokens", 0),
            vocab_size=data.get("vocab_size", 0),
        )

    async def get_next_token_logits(
        self,
        messages: List[Dict],
        continuation: str = "",
        continuation_ids: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Get raw logits for the next token position (Modal only).

        Returns (logits_array, vocab_size) where logits_array has shape (vocab_size,).
        Use this for step-by-step custom sampling loops.
        """
        if self.config["type"] != "modal":
            raise RuntimeError("get_next_token_logits requires Modal deployment mode")

        payload: Dict = {"messages": messages}
        if continuation_ids is not None:
            payload["continuation_ids"] = continuation_ids
        elif continuation:
            payload["continuation"] = continuation

        async with self._semaphore:
            url = f"{self.config['base_url'].rstrip('/')}/next_token_logits"
            data = await self._modal_post_with_retry(url, payload)

        logits = np.frombuffer(
            base64.b64decode(data["logits_b64"]), dtype=np.float32
        ).reshape(data["logits_shape"])

        return logits, data.get("vocab_size", logits.shape[0])

    # ── Response parsers ─────────────────────────────────────────────────

    def _parse_openai_response(self, data: Dict) -> Tuple[str, int]:
        """Parse OpenAI-compatible response format."""
        message = data["choices"][0]["message"]
        content = message.get("content", "") or ""
        reasoning = message.get("reasoning", "") or ""
        tokens = data.get("usage", {}).get("total_tokens", 0)

        # Strip inline <think>...</think> blocks if present
        stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Fallback to reasoning field if content is empty
        if not stripped and reasoning:
            stripped = reasoning.strip()

        return stripped or content.strip(), tokens

    def _parse_native_response(self, data: Dict) -> Tuple[str, int]:
        """Parse native Ollama /api/chat response format."""
        raw_content = data.get("message", {}).get("content", "") or ""
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        eval_tokens = data.get("eval_count", 0) or 0
        tokens = prompt_tokens + eval_tokens

        # Strip inline <think>...</think> blocks if present
        content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        # If stripping removed everything, extract from inside the think block
        if not content and "<think>" in raw_content:
            think_match = re.search(r"<think>(.*)</think>", raw_content, flags=re.DOTALL)
            if think_match:
                content = think_match.group(1).strip()

        logger.info("parse_native", raw_len=len(raw_content), result_len=len(content), preview=content[:150] if content else "(empty)")

        return content or raw_content.strip(), tokens

    async def parallel_calls(
        self,
        calls: List[Tuple[List[Dict], str, Dict]]
    ) -> List[Tuple[str, int]]:
        """Make multiple LLM calls in parallel"""
        tasks = []
        for messages, model, overrides in calls:
            task = self.call_llm(messages, model, **overrides)
            tasks.append(task)

        return await asyncio.gather(*tasks)

# Global instance
llm_client = LLMClient()
