#!/usr/bin/env python
"""Test-script for å sjekke om retrieval_graph og shared-moduler kan importeres."""

import sys
import os

# Legg til prosjektets rotmappe i sys.path
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)

# Legg til src-mappen i sys.path for å støtte begge importstiler
sys.path.append(os.path.join(project_root, "src"))

# Sett opp logging
from src.logger import setup_logger
logger = setup_logger("test-import")

# Test import av retrieval_graph
try:
    from retrieval_graph import graph
    logger.info("retrieval_graph importert vellykket!")
except ImportError as e:
    logger.error(f"Kunne ikke importere retrieval_graph: {str(e)}")

# Test import av shared
try:
    from shared import retrieval
    logger.info("shared.retrieval importert vellykket!")
except ImportError as e:
    logger.error(f"Kunne ikke importere shared.retrieval: {str(e)}")

# Test alternativ import av shared
try:
    from src.shared import retrieval as src_retrieval
    logger.info("src.shared.retrieval importert vellykket!")
except ImportError as e:
    logger.error(f"Kunne ikke importere src.shared.retrieval: {str(e)}")
