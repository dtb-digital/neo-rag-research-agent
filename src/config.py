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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lovdata-embedding-index")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Støttede modeller for konfigurasjon
SUPPORTED_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ],
    "anthropic": [
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
}

class SearchKwargs(TypedDict):
    """Søkekonfigurasjon."""
    k: int


@dataclass(kw_only=True)  
class AgentConfiguration:
    """Konfigurasjon for Neo RAG Research Agent."""

    # Modeller
    query_model: str = field(
        default="anthropic/claude-3-7-sonnet-20250219",
        metadata={
            "description": "Språkmodell for spørsmålsbehandling og agent-logikk.",
            "format": "provider/model-name",
            "supported_providers": ["openai", "anthropic"],
            "examples": [
                "anthropic/claude-3-7-sonnet-20250219",
                "anthropic/claude-sonnet-4-20250514",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-sonnet-20241022",
                "anthropic/claude-3-5-haiku-20241022"
            ]
        },
    )
    
    response_model: str = field(
        default="anthropic/claude-3-7-sonnet-20250219", 
        metadata={
            "description": "Språkmodell for respons-generering.",
            "format": "provider/model-name", 
            "supported_providers": ["openai", "anthropic"],
            "examples": [
                "anthropic/claude-3-7-sonnet-20250219",
                "anthropic/claude-sonnet-4-20250514",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-sonnet-20241022",
                "anthropic/claude-3-5-haiku-20241022"
            ]
        },
    )

    embedding_model: str = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Modell for tekstembedding (kun OpenAI støttes).",
            "format": "provider/model-name",
            "supported_providers": ["openai"],
            "examples": [
                "openai/text-embedding-3-small",
                "openai/text-embedding-3-large"
            ]
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


def validate_model_config(config: AgentConfiguration) -> tuple[bool, list[str]]:
    """Valider at modellkonfigurasjonen er gyldig og at nødvendige API-nøkler er tilgjengelige.
    
    Args:
        config: AgentConfiguration som skal valideres
        
    Returns:
        tuple: (er_gyldig, liste_med_feilmeldinger)
    """
    errors = []
    
    # Sjekk at modeller har riktig format
    for model_field in ["query_model", "response_model", "embedding_model"]:
        model_name = getattr(config, model_field)
        if "/" not in model_name:
            errors.append(f"{model_field} må være i format 'provider/model-name', fikk: {model_name}")
            continue
            
        provider, model = model_name.split("/", 1)
        
        # Valider provider
        if provider not in SUPPORTED_MODELS:
            errors.append(f"Ukjent provider '{provider}' for {model_field}. Støttede: {list(SUPPORTED_MODELS.keys())}")
            continue
            
        # Valider at model eksisterer for provider
        if model not in SUPPORTED_MODELS[provider]:
            errors.append(f"Ukjent modell '{model}' for provider '{provider}'. Støttede: {SUPPORTED_MODELS[provider]}")
            
        # Sjekk at API-nøkkel er tilgjengelig for provider
        if provider == "openai" and not OPENAI_API_KEY:
            errors.append(f"OPENAI_API_KEY er ikke satt, men {model_field} bruker OpenAI")
        elif provider == "anthropic" and not ANTHROPIC_API_KEY:
            errors.append(f"ANTHROPIC_API_KEY er ikke satt, men {model_field} bruker Anthropic")
    
    # Spesiell validering for embedding (kun OpenAI støttes)
    embedding_provider = config.embedding_model.split("/")[0]
    if embedding_provider != "openai":
        errors.append("embedding_model støtter kun OpenAI-modeller")
    
    return len(errors) == 0, errors


def get_available_models() -> dict[str, list[str]]:
    """Returner tilgjengelige modeller per provider.
    
    Returns:
        dict: Dictionary med provider som nøkkel og liste med modeller som verdi
    """
    return SUPPORTED_MODELS.copy()


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
        "ANTHROPIC_API_KEY": sensurert_verdi("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
        "LOG_LEVEL": LOG_LEVEL,
    }

# Valider konfigurasjon ved import
config_valid, config_errors = validate_config() 