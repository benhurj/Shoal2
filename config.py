# config.py
LLM_BASE_URL = "http://localhost:8080/v1"
LLM_MODEL = "qwen2.5-7b-instruct"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1          # Low temp for reliability
REACT_MAX_ITERATIONS = 6       # Safety limit