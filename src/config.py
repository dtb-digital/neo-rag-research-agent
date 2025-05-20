"""
Konfigurasjon og miljøvariabel-håndtering for Lovdata RAG-agent.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Last inn miljøvariabler fra .env-fil
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)

# API-nøkler
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone konfigurasjon
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lovdata-index")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

# Embedding-konfigurasjon
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")

# MCP konfigurasjon
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")

def validate_config() -> tuple[bool, list[str]]:
    """
    Validerer at alle nødvendige konfigurasjonsvariabler er satt.
    
    Returns:
        tuple: (er_gyldig, liste_med_feilmeldinger)
    """
    errors = []
    
    # Sjekk påkrevde API-nøkler
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
        "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION,
        "LOG_LEVEL": LOG_LEVEL,
        "DEBUG": DEBUG,
        "MCP_TRANSPORT": MCP_TRANSPORT,
    }

# Valider konfigurasjon ved import
config_valid, config_errors = validate_config() 