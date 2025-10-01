# app/llm_config.py

from langchain_ollama import Ollama
from langchain_openai import OpenAI  # or whatever the library is

def get_llm(provider: str, **kwargs):
    if provider == "ollama":
        return Ollama(model=kwargs.get("model_name", "llama3.1"), temperature=kwargs.get("temperature", 0))
    elif provider == "openai":
        return OpenAI(temperature=kwargs.get("temperature", 0), model=kwargs.get("model_name", "gpt-4"))
    else:
        raise ValueError(f"Unknown provider {provider}")
