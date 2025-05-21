"""Logging-konfigurasjon for prosjektet.

Dette modulet konfigurerer logging for hele prosjektet, og sikrer at all logging
blir fanget opp og skrevet til både konsoll og fil.
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# Importer konfigurasjon for å bruke LOG_LEVEL og DEBUG
try:
    from src.config import LOG_LEVEL, DEBUG
except ImportError:
    # Fallback hvis vi ikke kan importere fra src.config
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")

# Definer log levels for enkel konvertering fra streng
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Opprett logs-mappen hvis den ikke finnes
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Logger filnavn
log_file = os.path.join(logs_dir, "rag-research-agent.log")

# Formatteringer
if DEBUG:
    # Mer detaljert format for debugging
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
else:
    # Enklere format for produksjon
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Mer detaljert format for filen uansett
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def configure_logging():
    """Konfigurer logging for applikasjonen.
    
    Denne funksjonen setter opp loggingen for hele applikasjonen. Funksjonen vil kun
    konfigurere loggingen hvis det ikke allerede er gjort (for å unngå duplikasjon).
    """
    # Konfigurer root logger
    root_logger = logging.getLogger()
    
    # Sjekk om root loggeren allerede har handlere
    if root_logger.hasHandlers():
        # Loggeren har allerede handlere, vi fjerner dem for å unngå duplikasjon
        root_logger.handlers.clear()
    
    # Sett DEBUG nivå for å tillate all logging
    root_logger.setLevel(logging.DEBUG)
    
    # Oppsett for konsollhandler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(LOG_LEVELS.get(LOG_LEVEL.lower(), logging.INFO))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Oppsett for filhandler med rotasjon
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10 MB maks størrelse, behold 5 tidligere logger
    )
    file_handler.setLevel(logging.DEBUG)  # Logg alt til filen
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Demp for verbose tredjeparts loggere
    for logger_name in ["openai", "httpx", "httpcore", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logging.info("Logging konfigurert i shared/logging_config.py")

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Konfigurerer og returnerer en logger med spesifisert navn og nivå.
    Bruker den sentrale loggingskonfigurasjonen men tillater spesifikt nivå.
    
    Args:
        name: Navn på loggeren
        level: Log-nivå (debug, info, warning, error, critical)
        
    Returns:
        logging.Logger: Konfigurert logger
    """
    # Sikre at grunnleggende logging er konfigurert
    configure_logging()
    
    # Opprett logger med spesifisert navn
    logger = logging.getLogger(name)
    
    # Sett spesifisert nivå om gitt
    if level:
        log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
        logger.setLevel(log_level)
    
    return logger

def get_logger(name):
    """Få en logger med gitt navn.
    
    Args:
        name (str): Navnet på loggeren.
        
    Returns:
        Logger: En logger-instans med gitt navn.
    """
    return setup_logger(name)

# Pre-konfigurer default logger for lovdata-rag
logger = setup_logger("lovdata-rag") 