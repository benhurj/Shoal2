# llm_client.py
import asyncio
import re
import httpx
import structlog
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
from env_config import get_llm_config
from config import LLM_MAX_TOKENS, MAX_CONCURRENT_LLM_CALLS

logger = structlog.get_logger()

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
        async with self._semaphore:
            if self.config["type"] == "ollama_cloud":
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

        try:
            resp = await client.post(
                f"{self.config['base_url']}/v1/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(f"LLM call timed out (Ollama Cloud, model {model})")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM returned HTTP {e.response.status_code} (Ollama Cloud)")

        return self._parse_openai_response(resp.json())

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
