"""Felles hjelpefunksjoner for Neo RAG Research Agent."""

import os
from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def load_chat_model(fully_specified_name: str, **kwargs: Any) -> BaseChatModel:
    """Last inn en chat-modell fra et fullt spesifisert navn.

    Args:
        fully_specified_name (str): Streng i formatet 'provider/model'.
        **kwargs: Ekstra argumenter som sendes til modell-initialisering
        
    Returns:
        BaseChatModel: Initialisert chat-modell
        
    Raises:
        ValueError: Hvis modellformatet er ugyldig eller provider ikke støttes
        RuntimeError: Hvis nødvendige API-nøkler mangler
    """
    if "/" not in fully_specified_name:
        raise ValueError(f"Modellnavn må være i format 'provider/model', fikk: {fully_specified_name}")
    
    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    # Valider provider og API-nøkler
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY er ikke satt i miljøvariablene")
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY er ikke satt i miljøvariablene")
    else:
        raise ValueError(f"Ukjent provider: {provider}. Støttede providers: openai, anthropic")
    
    # Initialiser modell med langchain init_chat_model
    try:
        return init_chat_model(
            model, 
            model_provider=provider,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Kunne ikke initialisere {provider}/{model}: {str(e)}") from e


def get_model_info(fully_specified_name: str) -> Dict[str, str]:
    """Hent informasjon om en modell fra dens fullt spesifiserte navn.
    
    Args:
        fully_specified_name (str): Streng i formatet 'provider/model'
        
    Returns:
        Dict[str, str]: Dictionary med 'provider' og 'model' nøkler
        
    Raises:
        ValueError: Hvis modellformatet er ugyldig
    """
    if "/" not in fully_specified_name:
        raise ValueError(f"Modellnavn må være i format 'provider/model', fikk: {fully_specified_name}")
    
    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    return {
        "provider": provider,
        "model": model,
        "full_name": fully_specified_name
    } 