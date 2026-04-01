# modal_server.py
# Deploy: uv run modal deploy modal_server.py
# Requires: modal secret create huggingface-secret HF_TOKEN=hf_xxx
#
# Endpoints:
#   POST /query              — run full Shoal pipeline, returns AgentResponse
#   POST /next_token_logits  — raw next-token logits (for custom sampling loops)
#   GET  /health             — liveness check

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import modal

# ── Configuration ─────────────────────────────────────────────────────────────

WORKER_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
COMPILER_MODEL = "Qwen/Qwen3.5-9B"
WORKER_POOL_SIZE = 3  # K model copies; each gets its own CUDA stream

app = modal.App("shoal-dual")

# Project source files are added to the image at /app (copy=False → mounted at runtime)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.49.0",
        "numpy",
        "accelerate",
        "fastapi[standard]",
        "structlog",
        "pydantic",
        "python-dotenv",
        "httpx",
        "ddgs>=9.11.1",
        "mcp>=1.0.0",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[
            ".git", ".venv", "__pycache__", ".pytest_cache",
            "node_modules", "tests", ".env", "*.pyc", "*.pyo",
            "uv.lock", ".claude",
        ],
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


# ── Shared model-loading helpers ──────────────────────────────────────────────

def _load_model(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[shoal] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[shoal] Loading model {model_name} (float16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"[shoal] Loaded {model_name}. vocab_size={vocab_size}, device={model.device}")
    return model, tokenizer, vocab_size


def _apply_chat_template_safe(tokenizer, messages, **kwargs):
    """Apply chat template, stripping unsupported kwargs for non-Qwen tokenizers."""
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


# ── Generation primitive (module-level so InProcessLLMClient can call it) ─────

class _Msg:
    """Minimal message object with .model_dump() for _compute_generate."""
    __slots__ = ("role", "content")

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def model_dump(self):
        return {"role": self.role, "content": self.content}


class _Body:
    """Minimal request struct that _compute_generate reads."""
    __slots__ = ("messages", "max_tokens", "temperature", "top_p", "return_logits", "stop")

    def __init__(self, messages, max_tokens, temperature, top_p, stop):
        self.messages = messages
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.return_logits = False
        self.stop = stop


def _compute_generate(model, tokenizer, vocab_size, body):
    """Standard generation. Safe to run in a ThreadPoolExecutor thread."""
    import torch

    messages = [m.model_dump() for m in body.messages]
    input_text = _apply_chat_template_safe(
        tokenizer, messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=body.max_tokens,
        do_sample=body.temperature > 0,
        return_dict_in_generate=True,
    )
    if body.temperature > 0:
        gen_kwargs["temperature"] = body.temperature
        gen_kwargs["top_p"] = body.top_p

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated_ids = outputs.sequences[0][input_len:]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if body.stop:
        for s in body.stop:
            idx = content.find(s)
            if idx != -1:
                content = content[:idx]

    return {
        "content": content,
        "usage": {
            "prompt_tokens": int(input_len),
            "completion_tokens": len(generated_ids),
            "total_tokens": int(input_len) + len(generated_ids),
        },
        "vocab_size": int(vocab_size),
    }


# ── Worker pool ───────────────────────────────────────────────────────────────

class WorkerPool:
    """K copies of the worker model, each on its own CUDA stream."""

    def __init__(self, model_name, k):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[shoal] Initializing WorkerPool: {k} copies of {model_name}")
        self.k = k
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = None
        self.models = []
        self.streams = []

        for i in range(k):
            print(f"[shoal] Loading worker copy {i}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model.eval()
            if self.vocab_size is None:
                self.vocab_size = model.config.vocab_size
            self.models.append(model)
            self.streams.append(torch.cuda.Stream())
            mem_gb = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"[shoal] Worker copy {i} loaded. VRAM used so far: {mem_gb:.2f} GB")

        self.executor = ThreadPoolExecutor(max_workers=k)
        print(f"[shoal] WorkerPool ready. vocab_size={self.vocab_size}, k={k}")


# ── In-process LLM client (replaces HTTP calls inside the pipeline) ────────────

class InProcessLLMClient:
    """Drop-in for agent_hybrid.llm_client — calls loaded models directly in-process.

    Routes worker_idx calls to the WorkerPool (parallel CUDA streams).
    Serializes compiler calls with a threading.Semaphore so only one
    Qwen3.5-9B generate() runs at a time.
    """

    def __init__(self, models, worker_pool):
        # models: {model_name: (model, tokenizer, vocab_size)}
        self.models = models
        self.worker_pool = worker_pool
        self._executor = ThreadPoolExecutor(max_workers=20)
        self._compiler_lock = threading.Semaphore(1)

    async def call_llm(self, messages, model, stop=None, **overrides):
        import asyncio
        import re

        worker_idx = overrides.pop("worker_idx", None)
        overrides.pop("port", None)  # irrelevant in-process

        max_tokens = overrides.get("max_tokens", 512)
        temperature = overrides.get("temperature", 0.5)
        top_p = overrides.get("top_p", 1.0)

        msgs = [_Msg(m["role"], m["content"]) for m in messages]
        body = _Body(msgs, max_tokens, temperature, top_p, stop)

        loop = asyncio.get_running_loop()

        if worker_idx is not None and self.worker_pool is not None:
            pool_model = self.worker_pool.models[worker_idx]
            pool_stream = self.worker_pool.streams[worker_idx]
            pool_tokenizer = self.worker_pool.tokenizer
            pool_vocab_size = self.worker_pool.vocab_size

            def _run_worker():
                import torch
                with torch.cuda.stream(pool_stream):
                    return _compute_generate(pool_model, pool_tokenizer, pool_vocab_size, body)

            result = await loop.run_in_executor(self.worker_pool.executor, _run_worker)
        else:
            m_obj, tok, vs = self.models[model]

            def _run_compiler():
                with self._compiler_lock:
                    return _compute_generate(m_obj, tok, vs, body)

            result = await loop.run_in_executor(self._executor, _run_compiler)

        content = result["content"]
        # Strip Qwen3 thinking blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip() or content
        tokens = result["usage"]["total_tokens"]
        return content, tokens


# ── FastAPI application ───────────────────────────────────────────────────────

def _build_app(models, worker_pool, run_pipeline):
    """Build the FastAPI app.

    Args:
        models:       {model_name: (model, tokenizer, vocab_size)}
        worker_pool:  WorkerPool instance
        run_pipeline: async (query: str) -> AgentResponse
    """
    import asyncio
    import base64
    import torch
    import numpy as np
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List, Optional

    class QueryRequest(BaseModel):
        query: str

    class Message(BaseModel):
        role: str = "user"
        content: str = ""

    class GenerateRequest(BaseModel):
        model: str = WORKER_MODEL
        messages: List[Message]
        max_tokens: int = 512
        temperature: float = 0.5
        top_p: float = 1.0
        stop: Optional[List[str]] = None
        worker_idx: Optional[int] = None
        return_logits: bool = False

    class NextTokenRequest(BaseModel):
        model: str = WORKER_MODEL
        messages: List[Message]
        continuation: str = ""
        continuation_ids: Optional[List[int]] = None

    web_app = FastAPI(title="Shoal — Multi-Agent Ensemble")

    def _resolve(model_name):
        if model_name not in models:
            raise HTTPException(400, f"Unknown model: {model_name}. Available: {list(models.keys())}")
        return models[model_name]

    # ── Pipeline endpoint ─────────────────────────────────────────────────────

    @web_app.post("/query")
    async def query_endpoint(body: QueryRequest):
        """Run the full Shoal pipeline.

        Stages:
          1. Planner + Role Designer  (1× Qwen3.5-9B)
          2. Plan Adjuster            (1× Qwen3.5-9B)
          3. K×N executor samples     (SmolLM2-135M, K CUDA streams)
          4. K best-of-N evaluators   (1× Qwen3.5-9B each)
          5. Compiler                 (1× Qwen3.5-9B)
        """
        result = await run_pipeline(body.query)
        return JSONResponse(result.model_dump())

    # ── Per-model generation endpoint (used by local server in modal mode) ───────

    @web_app.post("/generate")
    async def generate_endpoint(body: GenerateRequest):
        """Single-model text generation. Routes worker_idx calls to the WorkerPool."""
        import asyncio, torch
        loop = asyncio.get_running_loop()

        if body.worker_idx is not None and worker_pool is not None:
            idx = body.worker_idx % worker_pool.k
            pool_model = worker_pool.models[idx]
            pool_stream = worker_pool.streams[idx]
            pool_tokenizer = worker_pool.tokenizer
            pool_vocab_size = worker_pool.vocab_size

            def _run_worker():
                with torch.cuda.stream(pool_stream):
                    return _compute_generate(pool_model, pool_tokenizer, pool_vocab_size, body)

            result = await loop.run_in_executor(worker_pool.executor, _run_worker)
        else:
            m_obj, tok, vs = _resolve(body.model)

            def _run_compiler():
                return _compute_generate(m_obj, tok, vs, body)

            result = await loop.run_in_executor(None, _run_compiler)

        return JSONResponse(result)

    # ── Token logits endpoint ─────────────────────────────────────────────────

    @web_app.post("/next_token_logits")
    async def next_token_logits(body: NextTokenRequest):
        """Return raw pre-softmax logits for the next token position."""
        model_name = body.model
        model, tokenizer, vocab_size = _resolve(model_name)

        messages = [m.model_dump() for m in body.messages]
        input_text = _apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        if body.continuation_ids is not None:
            base_ids = tokenizer(input_text, return_tensors="pt").input_ids
            cont_ids = torch.tensor([body.continuation_ids], dtype=base_ids.dtype)
            input_ids = torch.cat([base_ids, cont_ids], dim=1).to(model.device)
            model_inputs = {"input_ids": input_ids}
        else:
            if body.continuation:
                input_text += body.continuation
            model_inputs = tokenizer(input_text, return_tensors="pt")
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = model(**model_inputs)

        logits = outputs.logits[0, -1].cpu().float().numpy()
        return JSONResponse({
            "logits_b64": base64.b64encode(logits.astype(np.float32).tobytes()).decode("ascii"),
            "logits_shape": [int(logits.shape[0])],
            "vocab_size": int(vocab_size),
        })

    # ── Health ────────────────────────────────────────────────────────────────

    @web_app.get("/health")
    async def health():
        pool_info = None
        if worker_pool is not None:
            pool_info = {
                "worker_copies": worker_pool.k,
                "vocab_size": int(worker_pool.vocab_size),
            }
        import agent_hybrid as _ah
        mcp_tools = None
        if _ah.mcp_client is not None:
            try:
                mcp_tools = [t["name"] for t in await _ah.mcp_client.list_tools()]
            except Exception:
                mcp_tools = "error"
        return {
            "status": "ok",
            "models": {name: int(vs) for name, (_, _, vs) in models.items()},
            "worker_pool": pool_info,
            "mcp_tools": mcp_tools,
        }

    return web_app


# ── Single L4 container: compiler + K worker copies ───────────────────────────

@app.cls(
    gpu="L4",
    image=image,
    memory=32768,
    scaledown_window=3600,
    timeout=1200,
    max_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
class ShoalDualModel:

    @modal.enter()
    async def load_models(self):
        import asyncio
        import torch

        # Make project source files importable
        sys.path.insert(0, "/app")

        # Load compiler model (Qwen3.5-9B)
        c_model, c_tok, c_vs = _load_model(COMPILER_MODEL)

        # Worker pool — K copies of SmolLM2-135M for parallel follow-up decisions
        self.worker_pool = WorkerPool(WORKER_MODEL, k=WORKER_POOL_SIZE)
        hf_cache.commit()

        self.models = {
            COMPILER_MODEL: (c_model, c_tok, c_vs),
            WORKER_MODEL: (
                self.worker_pool.models[0],
                self.worker_pool.tokenizer,
                self.worker_pool.vocab_size,
            ),
        }

        # Create in-process LLM client and inject into agent_hybrid
        self._inprocess_client = InProcessLLMClient(self.models, self.worker_pool)
        import agent_hybrid
        agent_hybrid.llm_client = self._inprocess_client
        # Reduce pipeline load for single-L4 deployment (fit within 1200s timeout)
        agent_hybrid.ENSEMBLE_K = WORKER_POOL_SIZE
        agent_hybrid.SAMPLES_PER_ROLE = 1          # N=1; follow-ups replace N>1
        agent_hybrid.REACT_MAX_ITERATIONS = 2
        agent_hybrid.WORKER_POOL_SIZE = WORKER_POOL_SIZE
        agent_hybrid.MAX_FOLLOW_UPS = 2
        agent_hybrid.FOLLOW_UP_MAX_TOKENS = 64
        self._run_pipeline = agent_hybrid.run_agent_many_samples

        # Start MCP tool server subprocess for richer tool descriptions
        from tools.mcp_client import MCPClient
        self._mcp_client = MCPClient("/app/tools/mcp_server.py")
        try:
            await self._mcp_client.start()
            # Warm the tool cache so first request doesn't block
            await self._mcp_client.list_tools()
            agent_hybrid.mcp_client = self._mcp_client
            print("[shoal] MCP tool server started.")
        except Exception as e:
            print(f"[shoal] MCP tool server failed to start (degraded mode): {e}")
            self._mcp_client = None

        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        used_vram = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"[shoal] All models loaded. VRAM: {used_vram:.2f} GB / {total_vram:.2f} GB")
        print(f"[shoal] Pipeline ready. Endpoint: POST /query")

    @modal.asgi_app()
    def serve(self):
        return _build_app(self.models, self.worker_pool, self._run_pipeline)
