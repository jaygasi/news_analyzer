# biomed_analyzer/analyzer/llm_services/__init__.py
from .base_llm import BaseLLM
from .gemini_llm import GeminiLLM
from ..config import GEMINI_API_KEY 

def get_llm_service(service_name: str = "gemini", **kwargs) -> BaseLLM:
    if service_name.lower() == "gemini":
        # GEMINI_API_KEY is loaded from config, kwargs can override if needed
        api_key_to_use = kwargs.get('api_key', GEMINI_API_KEY)
        return GeminiLLM(api_key=api_key_to_use)
    else:
        raise ValueError(f"Unsupported LLM service: {service_name}")