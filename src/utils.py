"""
Hjelpefunksjoner for Lovdata RAG-agent.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from shared.logging_config import logger


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Sikrer at en mappe eksisterer, og oppretter den om nødvendig.
    
    Args:
        directory: Mappenavn eller Path-objekt
        
    Returns:
        Path: Path-objekt til mappen
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        logger.info(f"Opprettet mappe: {dir_path}")
    return dir_path


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Laster inn JSON-data fra en fil.
    
    Args:
        file_path: Sti til JSON-filen
        
    Returns:
        Dict: JSON-data som dict
        
    Raises:
        FileNotFoundError: Hvis filen ikke finnes
        json.JSONDecodeError: Hvis filen ikke inneholder gyldig JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Filen finnes ikke: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Lagrer dict som JSON-fil.
    
    Args:
        data: Data som skal lagres
        file_path: Sti til hvor filen skal lagres
        indent: Innrykk for JSON-formatering
    """
    file_path = Path(file_path)
    
    # Sikre at mappen eksisterer
    ensure_directory(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logger.debug(f"Lagret JSON-data til: {file_path}")


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    Formatterer metadata til en lesbar streng.
    
    Args:
        metadata: Metadata-dict
        
    Returns:
        str: Formatert metadata-streng
    """
    return "\n".join([f"{key}: {value}" for key, value in metadata.items()])


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Forkorter tekst hvis den er lengre enn max_length.
    
    Args:
        text: Teksten som skal forkortes
        max_length: Maksimal lengde
        
    Returns:
        str: Forkortet tekst
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..." 