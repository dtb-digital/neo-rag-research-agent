"""
Logger-konfigurasjon for Lovdata RAG-agent.
"""
import logging
import sys
from typing import Optional

from src.config import LOG_LEVEL, DEBUG

# Definer log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Konfigurerer og returnerer en logger med spesifisert navn og nivå.
    
    Args:
        name: Navn på loggeren
        level: Log-nivå (debug, info, warning, error, critical)
        
    Returns:
        logging.Logger: Konfigurert logger
    """
    # Bruk spesifisert level eller default fra config
    log_level = LOG_LEVELS.get(level.lower() if level else LOG_LEVEL, logging.INFO)
    
    # Opprett logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Fjern eksisterende handlers for å unngå duplikater
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Opprett handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    
    # Definer format
    if DEBUG:
        # Mer detaljert format for debugging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
    else:
        # Enklere format for produksjon
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Opprett default logger
logger = setup_logger("lovdata-rag") 