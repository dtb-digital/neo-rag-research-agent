"""Logging-konfigurasjon for prosjektet.

Dette modulet konfigurerer logging for hele prosjektet, og sikrer at all logging
blir fanget opp og skrevet til både konsoll og fil.
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler

# Opprett logs-mappen hvis den ikke finnes
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Logger filnavn
log_file = os.path.join(logs_dir, "rag-research-agent.log")

# Formatteringer
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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
        # Loggeren har allerede handlere, sannsynligvis fra src/logger.py
        # Vi logger en advarsel, men gjør ikke noe for å unngå duplikasjon
        logging.warning("Root logger har allerede handlers. Hopper over nytt loggingoppsett.")
        return
    
    # Sett DEBUG nivå for å tillate all logging
    root_logger.setLevel(logging.DEBUG)
    
    # Oppsett for konsollhandler hvis ingen slik handler finnes
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
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

def get_logger(name):
    """Få en logger med gitt navn.
    
    Args:
        name (str): Navnet på loggeren.
        
    Returns:
        Logger: En logger-instans med gitt navn.
    """
    return logging.getLogger(name) 