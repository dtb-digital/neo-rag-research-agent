"""Felles hjelpefunksjoner for Neo RAG Research Agent."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Last inn en chat-modell fra et fullt spesifisert navn.

    Args:
        fully_specified_name (str): Streng i formatet 'provider/model'.
        
    Returns:
        BaseChatModel: Initialisert chat-modell
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider) 