#!/usr/bin/env python
"""
Testskript for MCP-serveren.

Dette skriptet verifiserer at MCP-serveren starter uten feil.
"""

import os
import sys
import subprocess
import time

# Sikre at vi kan importere fra src-mappen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import setup_logger

logger = setup_logger("mcp-test")

def test_mcp_server_startup():
    """Test at MCP-serveren starter opp uten feil."""
    
    logger.info("Starter MCP-server...")
    
    # Start MCP-serveren som en subprosess
    process = subprocess.Popen(
        ['python', 'src/mcp_server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Vent litt for at serveren skal starte
    time.sleep(2)
    
    # Sjekk om prosessen fortsatt kjører
    if process.poll() is None:
        logger.info("MCP-server startet vellykket og kjører.")
        
        # Avslutt MCP-serveren
        logger.info("Avslutter MCP-server...")
        process.terminate()
        process.wait(timeout=5)
        logger.info(f"MCP-server avsluttet med kode: {process.returncode}")
        return True
    else:
        # Serveren avsluttet uventet
        stdout, stderr = process.communicate()
        logger.error(f"MCP-server avsluttet uventet med kode: {process.returncode}")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        return False

if __name__ == "__main__":
    logger.info("Starter test av MCP-server oppstart")
    try:
        success = test_mcp_server_startup()
        if success:
            logger.info("Test vellykket: MCP-serveren starter opp uten feil")
            sys.exit(0)
        else:
            logger.error("Test feilet: MCP-serveren startet ikke opp korrekt")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test avbrutt av bruker")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Uventet feil: {str(e)}")
        sys.exit(1) 