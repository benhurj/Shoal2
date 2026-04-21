# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Deployment Mode ──
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "ollama_cloud")  # Options: "local", "ollama_cloud", "openai"

# Cloud API configuration (loaded from .env)
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY")
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://api.ollama.cloud")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ── Worker LLM (runs Executor + Evaluator) ──
WORKER_MODEL = "gemma3:4b"
if DEPLOYMENT_MODE == "ollama_cloud":
    # Cloud mode - use API endpoints instead of local ports
    WORKER_PORTS = ["cloud-worker"]  # Single cloud endpoint
    WORKER_API_URLS = [f"{OLLAMA_CLOUD_BASE_URL}/v1"]  # Cloud API URL
elif DEPLOYMENT_MODE == "openai":
    # OpenAI mode - use OpenAI API
    WORKER_PORTS = ["openai-worker"]
    WORKER_API_URLS = [f"{OPENAI_BASE_URL}"]
else:
    # Local mode - use local Ollama instances
    WORKER_PORTS = [11434]  # K=1 instance for testing
LLM_MAX_TOKENS = 4096

# ── Compiler LLM (runs Planner + Compiler) ──
COMPILER_MODEL = "gemma3:12b"
if DEPLOYMENT_MODE == "ollama_cloud":
    COMPILER_PORTS = ["cloud-compiler"]  # Single cloud endpoint
    COMPILER_API_URLS = [f"{OLLAMA_CLOUD_BASE_URL}/v1"]  # Cloud API URL
elif DEPLOYMENT_MODE == "openai":
    COMPILER_PORTS = ["openai-compiler"]
    COMPILER_API_URLS = [f"{OPENAI_BASE_URL}"]
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
ENSEMBLE_K = 5  # Number of parallel agent loops (matches 5 defined roles)

# ── Agent loop limits ──
REACT_MAX_ITERATIONS = 4       # Max tool loops per executor run
MAX_PLAN_STEPS = 5             # Cap on planner sub-tasks
MAX_EVALUATOR_RETRIES = 2     # Retries per sub-task before aborting
MAX_CONCURRENT_LLM_CALLS = 10 # Semaphore limit for concurrent LLM requests
