# env_config.py
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class CloudConfig:
    """Configuration for cloud LLM services"""

    @staticmethod
    def deployment_mode() -> str:
        from config import DEPLOYMENT_MODE
        return DEPLOYMENT_MODE

    @staticmethod
    def ollama_cloud_api_key() -> Optional[str]:
        return os.getenv("OLLAMA_CLOUD_API_KEY")

    @staticmethod
    def ollama_cloud_base_url() -> str:
        return os.getenv("OLLAMA_CLOUD_BASE_URL", "https://api.ollama.cloud")

    @staticmethod
    def openai_api_key() -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @staticmethod
    def openai_base_url() -> str:
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @staticmethod
    def together_api_key() -> Optional[str]:
        return os.getenv("TOGETHER_API_KEY")

    @staticmethod
    def together_base_url() -> str:
        return os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")

    @staticmethod
    def modal_endpoint_url() -> Optional[str]:
        return os.getenv("MODAL_ENDPOINT_URL")

def get_llm_config():
    """Returns the appropriate LLM configuration based on environment settings"""
    mode = CloudConfig.deployment_mode()

    if mode == "ollama_cloud" and CloudConfig.ollama_cloud_api_key():
        return {
            "type": "ollama_cloud",
            "api_key": CloudConfig.ollama_cloud_api_key(),
            "base_url": CloudConfig.ollama_cloud_base_url(),
        }
    elif mode == "openai" and CloudConfig.openai_api_key():
        return {
            "type": "openai",
            "api_key": CloudConfig.openai_api_key(),
            "base_url": CloudConfig.openai_base_url(),
        }
    elif mode == "together" and CloudConfig.together_api_key():
        return {
            "type": "together",
            "api_key": CloudConfig.together_api_key(),
            "base_url": CloudConfig.together_base_url(),
        }
    elif mode == "modal" and CloudConfig.modal_endpoint_url():
        return {
            "type": "modal",
            "base_url": CloudConfig.modal_endpoint_url(),
        }
    elif mode == "hybrid":
        # Worker: any OpenAI-compatible local server (Ollama or bitnet.cpp)
        # Compiler: Modal
        return {
            "type": "hybrid",
            "worker_url": os.getenv("LOCAL_WORKER_URL", "http://localhost:11434/v1"),
            "worker_api_key": os.getenv("LOCAL_WORKER_API_KEY", "none"),
            "modal_url": CloudConfig.modal_endpoint_url() or "",
        }
    else:
        return {
            "type": "local",
            "base_url": "http://127.0.0.1"
        }
