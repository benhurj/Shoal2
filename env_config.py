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
    else:
        return {
            "type": "local",
            "base_url": "http://127.0.0.1"
        }
