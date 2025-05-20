#!/usr/bin/env python3
"""
Debug-skript for Lovdata RAG-agent tester.

Dette skriptet kjører testene med detaljert logging og feilsøkingsinformasjon.
Det fokuserer spesielt på å finne årsaken til at MCP-serveren ikke fungerer.

Bruk:
    python debug_tests.py
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path

# Prosjektkonfigurasjon
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Konfigurer logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "debug_tests.log"))
    ]
)
logger = logging.getLogger("debug-tests")

def check_environment():
    """Sjekk miljøvariabler og prosjektoppsett."""
    logger.info("Sjekker miljøvariabler...")
    
    # Sjekk påkrevde miljøvariabler
    required_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Manglende miljøvariabler: {', '.join(missing_vars)}")
        logger.info("Prøver å laste fra .env-fil...")
        
        try:
            from dotenv import load_dotenv
            dotenv_path = os.path.join(PROJECT_ROOT, ".env")
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path=dotenv_path)
                logger.info(f"Miljøvariabler lastet fra {dotenv_path}")
                
                # Sjekk igjen etter innlasting
                missing_vars = [var for var in required_vars if not os.environ.get(var)]
                if missing_vars:
                    logger.error(f"Fortsatt manglende miljøvariabler etter innlasting: {', '.join(missing_vars)}")
                    return False
                else:
                    logger.info("Alle påkrevde miljøvariabler funnet etter innlasting av .env")
                    return True
            else:
                logger.error(f"Ingen .env-fil funnet på {dotenv_path}")
                return False
        except ImportError:
            logger.error("Kunne ikke importere python-dotenv, installerer...")
            subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"])
            logger.info("Prøv å kjøre skriptet på nytt")
            return False
    
    logger.info("Alle påkrevde miljøvariabler funnet")
    
    # Sjekk prosjektstruktur
    logger.info("Sjekker prosjektstruktur...")
    
    src_path = os.path.join(PROJECT_ROOT, "src")
    if not os.path.exists(src_path):
        logger.error(f"src-mappen finnes ikke: {src_path}")
        return False
    
    mcp_server_script = os.path.join(src_path, "mcp_server.py")
    if not os.path.exists(mcp_server_script):
        logger.error(f"MCP-server-skriptet finnes ikke: {mcp_server_script}")
        return False
    
    logger.info("Prosjektstruktur ser ut til å være i orden")
    return True

def run_pinecone_test():
    """Kjør Pinecone-testen direkte."""
    logger.info("Kjører Pinecone-test...")
    
    test_script = os.path.join(SCRIPT_DIR, "test_pinecone_simple.py")
    
    try:
        # Sett miljøvariabler for detaljert logging
        env = os.environ.copy()
        env["LOG_LEVEL"] = "DEBUG"
        
        process = subprocess.Popen(
            [sys.executable, test_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Les output i sanntid
        for line in process.stdout:
            print(line, end="")
        
        # Vent på at prosessen skal avslutte
        process.wait(timeout=60)
        
        # Les stderr hvis det finnes
        stderr = process.stderr.read()
        if stderr:
            logger.error(f"Pinecone-test stderr:\n{stderr}")
        
        return process.returncode == 0
    
    except Exception as e:
        logger.exception(f"Feil ved kjøring av Pinecone-test: {e}")
        return False

def run_mcp_test_with_debugging():
    """Kjør MCP-testen med utvidet debugging."""
    logger.info("Kjører MCP-test med utvidet debugging...")
    
    test_script = os.path.join(SCRIPT_DIR, "test_mcp_simple.py")
    
    try:
        # Sett miljøvariabler for detaljert logging
        env = os.environ.copy()
        env["LOG_LEVEL"] = "DEBUG"
        env["DEBUG"] = "true"
        
        process = subprocess.Popen(
            [sys.executable, test_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Les output i sanntid
        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()  # Sikre at output vises umiddelbart
        
        # Vent på at prosessen skal avslutte (med lengre timeout)
        try:
            process.wait(timeout=300)  # 5 minutter timeout
        except subprocess.TimeoutExpired:
            logger.error("MCP-test timeout etter 5 minutter, avbryter...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.error("Prosessen ville ikke avslutte, bruker kill...")
                process.kill()
            return False
        
        # Les stderr hvis det finnes
        stderr = process.stderr.read()
        if stderr:
            logger.error(f"MCP-test stderr:\n{stderr}")
        
        return process.returncode == 0
    
    except Exception as e:
        logger.exception(f"Feil ved kjøring av MCP-test: {e}")
        return False

def check_mcp_server_directly():
    """Start MCP-serveren direkte og test grunnleggende funksjonalitet."""
    logger.info("Tester MCP-serveren direkte...")
    
    mcp_server_script = os.path.join(PROJECT_ROOT, "src", "mcp_server.py")
    
    try:
        # Sett miljøvariabler for detaljert logging
        env = os.environ.copy()
        env["LOG_LEVEL"] = "DEBUG"
        env["DEBUG"] = "true"
        
        # Start MCP-serveren
        server_process = subprocess.Popen(
            [sys.executable, mcp_server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1  # Linjebufret
        )
        
        logger.info("MCP-server startet, venter på oppstart...")
        time.sleep(2)
        
        # Sjekk om serveren fortsatt kjører
        if server_process.poll() is not None:
            logger.error(f"MCP-server avsluttet umiddelbart med kode {server_process.returncode}")
            stderr = server_process.stderr.read()
            logger.error(f"MCP-server stderr:\n{stderr}")
            return False
        
        # Prøv en enkel forespørsel
        logger.info("Sender testforespørsel til MCP-serveren...")
        test_request = {
            "name": "semantic_search",
            "params": {
                "query": "test",
                "top_k": 1
            }
        }
        import json
        request_json = json.dumps(test_request) + "\n"
        server_process.stdin.write(request_json)
        server_process.stdin.flush()
        
        # Sett en timer for timeout
        start_time = time.time()
        timeout = 60  # sekunder
        
        # Vent på respons eller timeout
        response = None
        stderr_output = []
        
        logger.info("Venter på respons...")
        
        while time.time() - start_time < timeout and server_process.poll() is None:
            # Sjekk stderr for eventuelle feilmeldinger
            while True:
                stderr_line = server_process.stderr.readline().strip()
                if stderr_line:
                    stderr_output.append(stderr_line)
                    logger.warning(f"MCP stderr: {stderr_line}")
                else:
                    break
            
            # Sjekk om vi har fått en respons
            stdout_line = server_process.stdout.readline().strip()
            if stdout_line:
                logger.info(f"MCP response: {stdout_line}")
                try:
                    response = json.loads(stdout_line)
                    break
                except json.JSONDecodeError:
                    logger.warning(f"Ugyldig JSON fra MCP: {stdout_line}")
            
            time.sleep(0.1)
        
        # Stopp serveren
        logger.info("Stopper MCP-serveren...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("MCP-server svarte ikke på terminate, bruker kill...")
            server_process.kill()
        
        # Analyser resultatet
        if response is not None:
            logger.info("MCP-server svarte på forespørselen")
            logger.info(f"Respons: {json.dumps(response, indent=2)}")
            return True
        elif server_process.poll() is not None:
            logger.error(f"MCP-server krasjet med kode {server_process.returncode}")
            stderr = "\n".join(stderr_output)
            logger.error(f"MCP-server stderr:\n{stderr}")
            return False
        else:
            logger.error("MCP-server timeout, ingen respons mottatt")
            stderr = "\n".join(stderr_output)
            logger.error(f"MCP-server stderr:\n{stderr}")
            return False
        
    except Exception as e:
        logger.exception(f"Feil ved direkte testing av MCP-server: {e}")
        return False

def main():
    """Hovedfunksjon."""
    logger.info("Starter debugging av tester...")
    
    # Sjekk miljøet
    if not check_environment():
        logger.error("Miljøsjekk feilet, avbryter testing")
        return 1
    
    # Kjør Pinecone-test
    logger.info("=== PINECONE-TEST ===")
    pinecone_result = run_pinecone_test()
    logger.info(f"Pinecone-test resultat: {'BESTÅTT' if pinecone_result else 'FEILET'}")
    
    # Kjør direkte MCP-server-test
    logger.info("=== DIREKTE MCP-SERVER-TEST ===")
    direct_mcp_result = check_mcp_server_directly()
    logger.info(f"Direkte MCP-server-test resultat: {'BESTÅTT' if direct_mcp_result else 'FEILET'}")
    
    # Kjør MCP-test
    logger.info("=== MCP-TEST ===")
    mcp_result = run_mcp_test_with_debugging()
    logger.info(f"MCP-test resultat: {'BESTÅTT' if mcp_result else 'FEILET'}")
    
    # Oppsummering
    logger.info("=== TESTRESULTATER ===")
    logger.info(f"Pinecone-test: {'BESTÅTT' if pinecone_result else 'FEILET'}")
    logger.info(f"Direkte MCP-server-test: {'BESTÅTT' if direct_mcp_result else 'FEILET'}")
    logger.info(f"MCP-test: {'BESTÅTT' if mcp_result else 'FEILET'}")
    
    overall_result = all([pinecone_result, direct_mcp_result, mcp_result])
    logger.info(f"Samlet resultat: {'ALLE TESTER BESTÅTT' if overall_result else 'NOEN TESTER FEILET'}")
    
    return 0 if overall_result else 1

if __name__ == "__main__":
    sys.exit(main()) 