# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Deployment Mode ──
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "modal")
# Options: "local", "ollama_cloud", "openai", "together", "modal", "hybrid"
# "hybrid": small worker model runs locally via Ollama, compiler on Modal

# Cloud API configuration (loaded from .env)
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY")
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://api.ollama.cloud")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
MODAL_ENDPOINT_URL = os.getenv("MODAL_ENDPOINT_URL")  # single L4 container (both models)

# ── Logprobs configuration (Together.ai) ──
LOGPROBS_TOP_K = 5  # Number of top tokens to return logprobs for (0-20)

# ── Worker LLM (runs Executor) ──
# In hybrid mode the worker runs on a local Ollama server.
# LOCAL_WORKER_MODEL must match the model name registered in Ollama.
if DEPLOYMENT_MODE == "hybrid":
    WORKER_MODEL = os.getenv("LOCAL_WORKER_MODEL", "smollm2:135m-instruct")
else:
    WORKER_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

if DEPLOYMENT_MODE == "ollama_cloud":
    WORKER_PORTS = ["cloud-worker"]
    WORKER_API_URLS = [f"{OLLAMA_CLOUD_BASE_URL}/v1"]
elif DEPLOYMENT_MODE == "openai":
    WORKER_PORTS = ["openai-worker"]
    WORKER_API_URLS = [f"{OPENAI_BASE_URL}"]
elif DEPLOYMENT_MODE == "together":
    WORKER_PORTS = ["together-worker"]
    WORKER_API_URLS = [f"{TOGETHER_BASE_URL}"]
elif DEPLOYMENT_MODE == "modal":
    WORKER_PORTS = ["modal-worker"]
    WORKER_API_URLS = [f"{MODAL_ENDPOINT_URL}"]
elif DEPLOYMENT_MODE == "hybrid":
    # URL set via LOCAL_WORKER_URL; port/path handled by llm_client hybrid routing
    WORKER_PORTS = ["hybrid-worker"]
    WORKER_API_URLS = [os.getenv("LOCAL_WORKER_URL", "http://localhost:11434/v1")]
else:
    WORKER_PORTS = [11434]
LLM_MAX_TOKENS = 4096

# ── Compiler LLM (runs Planner + Compiler) ──
COMPILER_MODEL = "Qwen/Qwen3.5-9B"
if DEPLOYMENT_MODE == "ollama_cloud":
    COMPILER_PORTS = ["cloud-compiler"]  # Single cloud endpoint
    COMPILER_API_URLS = [f"{OLLAMA_CLOUD_BASE_URL}/v1"]  # Cloud API URL
elif DEPLOYMENT_MODE == "openai":
    COMPILER_PORTS = ["openai-compiler"]
    COMPILER_API_URLS = [f"{OPENAI_BASE_URL}"]
elif DEPLOYMENT_MODE == "together":
    COMPILER_PORTS = ["together-compiler"]
    COMPILER_API_URLS = [f"{TOGETHER_BASE_URL}"]
elif DEPLOYMENT_MODE == "modal":
    COMPILER_PORTS = ["modal-compiler"]
    COMPILER_API_URLS = [f"{MODAL_ENDPOINT_URL}"]
else:
    COMPILER_PORTS = [11440]  # L=1 instance
COMPILER_TEMPERATURE = 0.8  # Higher for creative planning/compilation
COMPILER_TOP_P = 0.95

# ── Stochastic sampling fallback (for overflow agents beyond defined roles) ──
TEMPERATURE_MEAN = 0.15
TEMPERATURE_STD = 0.05
TOP_P_MEAN = 0.85
TOP_P_STD = 0.05

# ── Ensemble configuration ──
ENSEMBLE_K = 3  # Number of parallel agent loops

# ── Many-samples architecture ──
SAMPLES_PER_ROLE = 5   # N samples per role (many-samples path)
WORKER_POOL_SIZE = 3   # K worker model copies; should match ENSEMBLE_K
MAX_FOLLOW_UPS = 2     # Max 135M follow-up decisions per plan step
FOLLOW_UP_MAX_TOKENS = 64  # Token budget for 135M decision output

# ── Per-stage token limits ──
PLANNER_MAX_TOKENS = 512
EXECUTOR_MAX_TOKENS = 512
SELF_CHECK_MAX_TOKENS = 16  # legacy: self-check (/agent endpoint only)
EVALUATOR_MAX_TOKENS = 512
COMPILER_MAX_TOKENS = 512

# ── Agent loop limits ──
REACT_MAX_ITERATIONS = 4       # Max tool loops per executor run
MAX_PLAN_STEPS = 3             # Cap on planner sub-tasks
MAX_AGENT_RETRIES = 1          # legacy: retry limit per agent (/agent endpoint)
MAX_CONCURRENT_LLM_CALLS = 15  # Semaphore limit; raised for K×N parallel calls

# ── Search tool ──
SEARCH_TOP_N = 3          # Default number of pages to fetch per search
SEARCH_MAX_CHARS = 4000   # Max characters per page (truncation guard)
