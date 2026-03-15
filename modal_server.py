# modal_server.py
# Deploy: modal deploy modal_server.py
# Requires: pip install modal && modal setup
# Requires: modal secret create huggingface-secret HF_TOKEN=hf_xxx
#
# After deploy, Modal prints one endpoint URL. Set it in .env:
#   MODAL_ENDPOINT_URL=https://<workspace>--shoal-dual-shoaldualmodel-serve.modal.run

import modal

# ── Configuration ──
WORKER_MODEL = "Qwen/Qwen3.5-0.8B"
COMPILER_MODEL = "Qwen/Qwen3.5-9B"

app = modal.App("shoal-dual")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.49.0",
        "numpy",
        "accelerate",
        "fastapi[standard]",
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


# ── Shared logic ─────────────────────────────────────────────────────

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
    print(f"[shoal] Model loaded. vocab_size={vocab_size}, device={model.device}")
    return model, tokenizer, vocab_size


def _build_app(models):
    """Build FastAPI app with multi-model routing.

    Args:
        models: dict mapping model_name -> (model, tokenizer, vocab_size)
    """
    import base64
    import torch
    import numpy as np
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any

    # ── Request/response schemas for Swagger ──────────────────────────

    class Message(BaseModel):
        role: str = "user"
        content: str = "Hello"

    class GenerateRequest(BaseModel):
        model: str = Field(default=WORKER_MODEL, description="Model name")
        messages: List[Message]
        max_tokens: int = 512
        temperature: float = 0.5
        top_p: float = 1.0
        return_logits: bool = False
        stop: Optional[List[str]] = None

    class BranchConfig(BaseModel):
        temperature: float = 0.5
        top_p: float = 1.0

    class BranchedRequest(BaseModel):
        model: str = Field(default=COMPILER_MODEL, description="Model name")
        messages: List[Message]
        branch_configs: List[BranchConfig]
        max_tokens: int = 512
        stop: Optional[List[str]] = None

    class NextTokenRequest(BaseModel):
        model: str = Field(default=WORKER_MODEL, description="Model name")
        messages: List[Message]
        continuation: str = ""
        continuation_ids: Optional[List[int]] = None

    web_app = FastAPI(title="Shoal Dual-Model Server")
    default_model = WORKER_MODEL

    def _resolve(model_name):
        if model_name not in models:
            raise HTTPException(400, f"Unknown model: {model_name}. Available: {list(models.keys())}")
        return models[model_name]

    @web_app.post("/generate")
    async def generate(body: GenerateRequest):
        model_name = body.model
        model, tokenizer, vocab_size = _resolve(model_name)

        messages = [m.model_dump() for m in body.messages]
        max_tokens = body.max_tokens
        temperature = body.temperature
        top_p = body.top_p
        return_logits = body.return_logits
        stop = body.stop

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            return_dict_in_generate=True,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if return_logits:
            gen_kwargs["output_logits"] = True

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        generated_ids = outputs.sequences[0][input_len:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if stop:
            for s in stop:
                idx = content.find(s)
                if idx != -1:
                    content = content[:idx]

        tokens = [tokenizer.decode([t]) for t in generated_ids]

        response = {
            "content": content,
            "tokens": tokens,
            "usage": {
                "prompt_tokens": int(input_len),
                "completion_tokens": len(generated_ids),
                "total_tokens": int(input_len) + len(generated_ids),
            },
            "vocab_size": int(vocab_size),
        }

        if return_logits and hasattr(outputs, "logits") and outputs.logits:
            logits_array = (
                torch.stack(outputs.logits).squeeze(1).cpu().float().numpy()
            )
            response["logits_b64"] = base64.b64encode(
                logits_array.astype(np.float32).tobytes()
            ).decode("ascii")
            response["logits_shape"] = list(logits_array.shape)

        return JSONResponse(response)

    @web_app.post("/next_token_logits")
    async def next_token_logits(body: NextTokenRequest):
        model_name = body.model
        model, tokenizer, vocab_size = _resolve(model_name)

        messages = [m.model_dump() for m in body.messages]
        continuation = body.continuation
        continuation_ids = body.continuation_ids

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        if continuation_ids is not None:
            base_ids = tokenizer(input_text, return_tensors="pt").input_ids
            cont_ids = torch.tensor([continuation_ids], dtype=base_ids.dtype)
            input_ids = torch.cat([base_ids, cont_ids], dim=1).to(model.device)
            model_inputs = {"input_ids": input_ids}
        else:
            if continuation:
                input_text += continuation
            model_inputs = tokenizer(input_text, return_tensors="pt")
            model_inputs = {
                k: v.to(model.device) for k, v in model_inputs.items()
            }

        with torch.no_grad():
            outputs = model(**model_inputs)

        logits = outputs.logits[0, -1].cpu().float().numpy()

        response = {
            "logits_b64": base64.b64encode(
                logits.astype(np.float32).tobytes()
            ).decode("ascii"),
            "logits_shape": [int(logits.shape[0])],
            "vocab_size": int(vocab_size),
        }
        return JSONResponse(response)

    # ── Branched generation ───────────────────────────────────────────

    def _top_p_sample(logits_1d, temperature, top_p):
        """Apply temperature scaling + nucleus (top-p) sampling to a 1-D logit tensor.
        Returns a single sampled token id (int).
        """
        if temperature <= 0:
            return int(logits_1d.argmax().item())

        scaled = logits_1d / temperature
        probs = torch.softmax(scaled, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Zero out tokens beyond the top-p threshold
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()

        chosen_pos = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[chosen_pos].item())

    @web_app.post("/generate_branched")
    async def generate_branched(body: BranchedRequest):
        model_name = body.model
        model, tokenizer, vocab_size = _resolve(model_name)

        messages = [m.model_dump() for m in body.messages]
        branch_configs = [bc.model_dump() for bc in body.branch_configs]
        max_tokens = body.max_tokens
        stop = body.stop

        if not branch_configs:
            raise HTTPException(400, "branch_configs is required and must be non-empty")

        K = len(branch_configs)

        # Tokenize prompt once
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        # Initial forward pass — get logits + KV cache for the shared prompt
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        logits = out.logits[0, -1]  # shape (vocab_size,)
        cache = out.past_key_values  # DynamicCache or tuple of (K, V) per layer

        # Per-branch state
        branch_token_ids = [[] for _ in range(K)]  # generated token ids
        branch_done = [False] * K
        eos_id = tokenizer.eos_token_id

        # Sample first token for each branch from shared logits
        for b, cfg in enumerate(branch_configs):
            tid = _top_p_sample(logits, cfg.get("temperature", 0.5), cfg.get("top_p", 1.0))
            branch_token_ids[b].append(tid)
            if tid == eos_id:
                branch_done[b] = True

        # Expand KV cache from batch=1 to batch=K
        # Qwen3.5 uses Qwen3_5DynamicCache — a hybrid cache with
        # key/value (attention), conv_states, and ssm_states (linear attn)
        print(f"[shoal] Cache type: {type(cache).__name__}, K={K}")

        # Expand key_cache / value_cache (attention layers)
        for layer_idx in range(len(cache.key_cache)):
            k = cache.key_cache[layer_idx]
            v = cache.value_cache[layer_idx]
            if k is not None:
                cache.key_cache[layer_idx] = k.repeat(K, 1, 1, 1)
            if v is not None:
                cache.value_cache[layer_idx] = v.repeat(K, 1, 1, 1)

        # Expand conv_states (linear attention layers — causal conv1d state)
        if hasattr(cache, 'conv_states'):
            for layer_idx in range(len(cache.conv_states)):
                cs = cache.conv_states[layer_idx]
                if cs is not None:
                    # conv_state shape: (batch, channels, conv_width)
                    cache.conv_states[layer_idx] = cs.repeat(K, 1, 1)

        # Expand ssm_states (linear attention layers — recurrent/SSM state)
        if hasattr(cache, 'ssm_states'):
            for layer_idx in range(len(cache.ssm_states)):
                ss = cache.ssm_states[layer_idx]
                if ss is not None:
                    # ssm_state shape: (batch, ...) — repeat along batch dim
                    repeat_dims = [K] + [1] * (ss.dim() - 1)
                    cache.ssm_states[layer_idx] = ss.repeat(*repeat_dims)

        print(f"[shoal] Cache expanded to batch={K}")

        # Autoregressive loop with batched forward passes
        for step in range(1, max_tokens):
            if all(branch_done):
                break

            # Build batched input: (K, 1) — last token per branch
            next_ids = torch.tensor(
                [[branch_token_ids[b][-1]] for b in range(K)],
                dtype=torch.long, device=model.device,
            )

            with torch.no_grad():
                out = model(input_ids=next_ids, past_key_values=cache, use_cache=True)

            cache = out.past_key_values
            all_logits = out.logits[:, -1, :]  # (K, vocab_size)

            for b, cfg in enumerate(branch_configs):
                if branch_done[b]:
                    continue
                tid = _top_p_sample(
                    all_logits[b], cfg.get("temperature", 0.5), cfg.get("top_p", 1.0)
                )
                branch_token_ids[b].append(tid)

                if tid == eos_id:
                    branch_done[b] = True
                    continue

                # Check stop sequences
                if stop:
                    decoded_tail = tokenizer.decode(branch_token_ids[b][-20:], skip_special_tokens=True)
                    for s in stop:
                        if s in decoded_tail:
                            branch_done[b] = True
                            break

        # Decode branches
        branches = []
        for b in range(K):
            content = tokenizer.decode(branch_token_ids[b], skip_special_tokens=True)
            # Trim at stop sequence
            if stop:
                for s in stop:
                    idx = content.find(s)
                    if idx != -1:
                        content = content[:idx]
            branches.append({
                "content": content,
                "tokens_used": len(branch_token_ids[b]),
            })

        response = {
            "branches": branches,
            "prompt_tokens": int(input_len),
            "vocab_size": int(vocab_size),
        }
        return JSONResponse(response)

    # ── Health ────────────────────────────────────────────────────────

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "models": {name: int(vs) for name, (_, _, vs) in models.items()},
        }

    return web_app


# ── Single L4 container with both models ─────────────────────────────

@app.cls(
    gpu="L4",
    image=image,
    memory=32768,
    scaledown_window=3600,
    timeout=600,
    max_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
class ShoalDualModel:

    @modal.enter()
    def load_models(self):
        w_model, w_tok, w_vs = _load_model(WORKER_MODEL)
        c_model, c_tok, c_vs = _load_model(COMPILER_MODEL)
        hf_cache.commit()
        self.models = {
            WORKER_MODEL: (w_model, w_tok, w_vs),
            COMPILER_MODEL: (c_model, c_tok, c_vs),
        }

    @modal.asgi_app()
    def serve(self):
        return _build_app(self.models)
