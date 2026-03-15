# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Deployment Mode ──
DEPLOYMENT_MODE = "modal"  # Options: "local", "ollama_cloud", "openai", "together", "modal"

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

# ── Worker LLM (runs Executor + Evaluator) ──
WORKER_MODEL = "Qwen/Qwen3.5-0.8B"
if DEPLOYMENT_MODE == "ollama_cloud":
    # Cloud mode - use API endpoints instead of local ports
    WORKER_PORTS = ["cloud-worker"]  # Single cloud endpoint
    WORKER_API_URLS = [f"{OLLAMA_CLOUD_BASE_URL}/v1"]  # Cloud API URL
elif DEPLOYMENT_MODE == "openai":
    # OpenAI mode - use OpenAI API
    WORKER_PORTS = ["openai-worker"]
    WORKER_API_URLS = [f"{OPENAI_BASE_URL}"]
elif DEPLOYMENT_MODE == "together":
    # Together.ai mode
    WORKER_PORTS = ["together-worker"]
    WORKER_API_URLS = [f"{TOGETHER_BASE_URL}"]
elif DEPLOYMENT_MODE == "modal":
    # Modal mode - single L4 container with both models
    WORKER_PORTS = ["modal-worker"]
    WORKER_API_URLS = [f"{MODAL_ENDPOINT_URL}"]
else:
    # Local mode - use local Ollama instances
    WORKER_PORTS = [11434]  # K=1 instance for testing
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

# ── Agent loop limits ──
REACT_MAX_ITERATIONS = 6       # Max tool loops per executor run
MAX_PLAN_STEPS = 5             # Cap on planner sub-tasks
MAX_EVALUATOR_RETRIES = 2     # Retries per sub-task before aborting
MAX_CONCURRENT_LLM_CALLS = 3 # Semaphore limit for concurrent LLM requests

# ── Search tool ──
SEARCH_TOP_N = 3          # Default number of pages to fetch per search
SEARCH_MAX_CHARS = 2000   # Max characters per page (truncation guard)
