# Shoal2 Architecture — Multi-Instance Stochastic Ensemble

## Full Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI :8000
    participant LLM as LLMClient (shared httpx + semaphore)
    participant OC as Ollama Cloud API

    U->>API: POST /agent {query}

    Note over API: Sample K stochastic configs<br/>temp ~ N(0.15, 0.05)<br/>top_p ~ N(0.85, 0.05)

    par K=5 Agent Loops (asyncio.gather)
        Note over LLM: Stage 1: Planner (gemma3:12b)
        API->>LLM: Agent 1–5 plan requests
        LLM->>OC: 5 concurrent /v1/chat/completions
        OC-->>LLM: Numbered plans
        LLM-->>API: Parsed plans

        Note over LLM: Stage 2: Executor (gemma3:4b)
        API->>LLM: Agent 1–5 ReAct loops
        LLM->>OC: Concurrent tool-use calls
        Note right of OC: Auto-scaled GPU fleet<br/>handles concurrent requests
        OC-->>LLM: Completions
        LLM-->>API: Executor results

        Note over LLM: Stage 3: Evaluator (gemma3:4b)
        API->>LLM: Agent 1–5 evaluations
        LLM->>OC: Concurrent eval calls
        OC-->>LLM: PASS/FAIL
        LLM-->>API: Evaluation results
    end

    Note over LLM: Stage 4: Compiler (gemma3:12b)
    API->>LLM: Compile 5 evaluated results
    LLM->>OC: /v1/chat/completions
    OC-->>LLM: Synthesized answer
    LLM-->>API: Compiled answer

    API-->>U: AgentResponse JSON
```

## Model Routing

```mermaid
flowchart LR
    subgraph CLIENT ["LLMClient (shared httpx.AsyncClient)"]
        SEM["asyncio.Semaphore(10)"]
        POOL["Connection Pool<br/>max=20, keepalive=10"]
    end
    subgraph CLOUD ["Ollama Cloud (auto-scaling)"]
        subgraph COMPILER_MODEL ["gemma3:12b"]
            P["Planner"]
            C["Compiler"]
        end
        subgraph WORKER_MODEL ["gemma3:4b"]
            E1["Executor 1"]
            E2["Executor 2"]
            E3["Executor 3"]
            E4["Executor 4"]
            E5["Executor 5"]
            EV1["Evaluator 1"]
            EV2["Evaluator 2"]
            EV3["Evaluator 3"]
            EV4["Evaluator 4"]
            EV5["Evaluator 5"]
        end
    end

    SEM --> POOL
    POOL -->|concurrent requests| P & C
    POOL -->|concurrent requests| E1 & E2 & E3 & E4 & E5
    P -->|plan| E1 & E2 & E3 & E4 & E5
    E1 --> EV1
    E2 --> EV2
    E3 --> EV3
    E4 --> EV4
    E5 --> EV5
    EV1 & EV2 & EV3 & EV4 & EV5 -->|evaluated results| C
```

## State & Memory

| Type | Component | Detail |
| :--- | :--- | :--- |
| **Stochastic Config** | `sample_ensemble_configs()` | Drawn fresh from N(μ,σ) each request |
| **Worker Memory** | `messages` list | Per-agent, per-sub-task, discarded after |
| **Ensemble State** | `EnsembleMemberResult` × K | Each carries execution trace + PASS/FAIL |
| **Planner** | `gemma3:12b` via Ollama Cloud | Decomposes query into sub-tasks |
| **Compiler** | `gemma3:12b` via Ollama Cloud | Synthesizes K results into best answer |
| **LLM Client** | `llm_client.py` | Shared httpx client, connection pooling, semaphore(10) |
| **Concurrency** | `asyncio.gather` + semaphore | K=5 loops truly parallel via cloud auto-scaling |
| **Statelessness** | `main.py` | No persistence between requests |
