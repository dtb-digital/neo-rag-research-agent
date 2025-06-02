"""Konfigurasjon for Neo RAG Research Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

# Last inn miljøvariabler fra .env-fil
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)

# API-nøkler og konfigurasjon
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lovdata-embedding-index")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class SearchKwargs(TypedDict):
    """Søkekonfigurasjon."""
    k: int


@dataclass(kw_only=True)  
class AgentConfiguration:
    """Konfigurasjon for Neo RAG Research Agent."""

    # Modeller
    query_model: str = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "Språkmodell for spørsmålsbehandling og agent-logikk. Format: provider/model-name."
        },
    )
    
    response_model: str = field(
        default="openai/gpt-4o-mini", 
        metadata={
            "description": "Språkmodell for respons-generering. Format: provider/model-name."
        },
    )

    embedding_model: str = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Modell for tekstembedding. Format: provider/model-name."
        },
    )

    # Søkekonfigurasjon
    search_kwargs: SearchKwargs = field(
        default_factory=lambda: {"k": 5},
        metadata={
            "description": "Konfigurasjon for søkeprosessen, inkludert antall dokumenter som skal hentes."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> AgentConfiguration:
        """Lag en konfigurasjon fra en RunnableConfig.
        
        Args:
            config: RunnableConfig med konfigurable parametre
            
        Returns:
            AgentConfiguration instans
        """
        if not config or "configurable" not in config:
            return cls()
        
        # Filtrer til kun gyldige AgentConfiguration parametere
        valid_params = {
            "query_model", "response_model", "embedding_model", "search_kwargs"
        }
        filtered_config = {
            k: v for k, v in config["configurable"].items()
            if k in valid_params
        }
        
        return cls(**filtered_config)


def validate_config() -> tuple[bool, list[str]]:
    """Valider at alle nødvendige konfigurasjonsvariabler er satt.
    
    Returns:
        tuple: (er_gyldig, liste_med_feilmeldinger)
    """
    errors = []
    
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY er ikke satt i miljøvariablene")
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY er ikke satt i miljøvariablene")
    
    return len(errors) == 0, errors

def get_config_dict() -> dict:
    """
    Returnerer alle konfigurasjonsvariabler som en dictionary.
    Nyttig for logging og debugging.
    
    Returns:
        dict: Konfigurasjonsverdier (med sensurerte API-nøkler)
    """
    # Sensurer API-nøkler i output
    def sensurert_verdi(key: str, value: Optional[str]) -> str:
        if value is None:
            return "None"
        if "API_KEY" in key and value:
            return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
        return value
    
    return {
        "PINECONE_API_KEY": sensurert_verdi("PINECONE_API_KEY", PINECONE_API_KEY),
        "OPENAI_API_KEY": sensurert_verdi("OPENAI_API_KEY", OPENAI_API_KEY),
        "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
        "LOG_LEVEL": LOG_LEVEL,
    }

# Valider konfigurasjon ved import
config_valid, config_errors = validate_config() 