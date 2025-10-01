# app/llm_config.py

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

def get_llm(llm_provider: str):
    """
    Returns an instance of the specified LLM.

    Args:
        llm_provider (str): The name of the LLM provider ('ollama' or 'openai').

    Returns:
        An instance of the LLM.
    """
    if llm_provider == "ollama":
        # Ensure Ollama is running and the model (e.g., 'llama3.1') is available
        return ChatOllama(model="llama3.1", temperature=0)
    elif llm_provider == "openai":
        # Ensure you have OPENAI_API_KEY set in your environment variables
        return ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
