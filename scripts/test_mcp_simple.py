#!/usr/bin/env python3
"""
Enkel test av MCP-serveren.

Dette skriptet tester at MCP-serveren starter og svarer korrekt på kommandoer.
Det kjører enkle tester for semantic_search og get_document, uten å gå gjennom
den fulle grafen.

Bruk:
    python test_mcp_simple.py
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from threading import Timer

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-test")

# Konfigurasjon
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MCP_SERVER_SCRIPT = os.path.join(PROJECT_ROOT, "src", "mcp_server.py")
DEFAULT_TIMEOUT = 60  # sekunder

def load_env_vars():
    """Last inn miljøvariabler fra .env-fil."""
    dotenv_path = os.path.join(PROJECT_ROOT, ".env")
    
    if os.path.exists(dotenv_path):
        logger.info(f"Laster miljøvariabler fra {dotenv_path}")
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=dotenv_path)
            return True
        except ImportError:
            logger.warning("python-dotenv ikke installert, miljøvariabler lastes ikke fra .env-fil")
            return False
    else:
        logger.warning(f"Ingen .env-fil funnet på {dotenv_path}")
        return False

class MCPServerTest:
    """Klasse for testing av MCP-serveren."""
    
    def __init__(self, timeout=DEFAULT_TIMEOUT, debug=False):
        """Initialiser testen."""
        self.timeout = timeout
        self.debug = debug
        self.process = None
        self.results = {}
    
    def start_server(self):
        """Start MCP-serveren."""
        logger.info("Starter MCP-serveren...")
        
        self.process = subprocess.Popen(
            [sys.executable, MCP_SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Linjebufret
            env=os.environ.copy()
        )
        
        # Gi serveren tid til å starte
        time.sleep(2)
        
        # Sjekk at serveren fortsatt kjører
        if self.process.poll() is not None:
            stderr = self.process.stderr.read()
            logger.error(f"MCP-server avsluttet med kode {self.process.returncode}")
            logger.error(f"stderr: {stderr}")
            return False
        
        logger.info("MCP-server startet")
        return True
    
    def stop_server(self):
        """Stopp MCP-serveren."""
        if self.process is not None:
            logger.info("Stopper MCP-serveren...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("MCP-serveren svarte ikke på terminate, bruker kill...")
                self.process.kill()
            logger.info("MCP-server stoppet")
    
    def call_tool(self, tool_name, params):
        """Kall et MCP-verktøy."""
        if self.process is None:
            raise RuntimeError("MCP-server kjører ikke")
            
        # Opprett MCP-forespørsel
        request = {
            "name": tool_name,
            "params": params
        }
        
        # Serialiser som JSON og send til serveren
        request_json = json.dumps(request)
        if self.debug:
            logger.debug(f"Sender: {request_json}")
        else:
            logger.info(f"Sender forespørsel til {tool_name} med parametre: {json.dumps(params)}")
            
        # Skriv til stdin og få respons
        self.process.stdin.write(request_json + "\n")
        self.process.stdin.flush()
        
        # Les responser fra stdout til vi finner en gyldig JSON
        line = None
        response = None
        start_time = time.time()
        
        # Opprett en timer for å avbryte hvis operasjonen tar for lang tid
        timeout_triggered = [False]
        
        def handle_timeout():
            timeout_triggered[0] = True
            logger.error(f"Timeout etter {self.timeout} sekunder ved kall til {tool_name}")
            # Vi kan ikke avbryte readline() direkte, så vi må terminere hele prosessen
            if self.process:
                logger.error("Terminerer MCP-server på grunn av timeout")
                try:
                    self.process.terminate()
                except:
                    pass

        # Sett opp timeout-timer
        timer = Timer(self.timeout, handle_timeout)
        timer.start()
        
        try:
            logger.info(f"Venter på svar fra {tool_name}...")
            
            while time.time() - start_time < self.timeout and not timeout_triggered[0]:
                # Sjekk om prosessen fortsatt kjører
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read() if self.process.stderr else "Ingen stderr"
                    logger.error(f"MCP-server avsluttet uventet med kode {self.process.returncode}")
                    logger.error(f"stderr: {stderr}")
                    raise RuntimeError(f"MCP-server avsluttet uventet med kode {self.process.returncode}")
                    
                # Les en linje fra stdout
                line = self.process.stdout.readline().strip()
                
                if not line:
                    # Kort pause for å ikke sluke CPU
                    time.sleep(0.1)
                    # Logg hver 5. sekund
                    elapsed = time.time() - start_time
                    if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                        logger.info(f"Venter fortsatt på svar... ({int(elapsed)}s)")
                    continue
                    
                if self.debug:
                    logger.debug(f"Mottok: {line}")
                else:
                    logger.info(f"Mottok svar fra serveren: {line[:100]}...")
                    
                try:
                    response = json.loads(line)
                    logger.info(f"Mottok gyldig JSON-svar fra {tool_name}")
                    break
                except json.JSONDecodeError:
                    if "error" in line.lower():
                        logger.error(f"Feilmelding fra server: {line}")
                        # Lag et feil-objekt for å returnere
                        response = {"error": line}
                        break
                    logger.warning(f"Ugyldig JSON: {line}")
                    continue
        except Exception as e:
            logger.error(f"Feil ved venting på svar: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Avbryt timeren hvis den fortsatt kjører
            timer.cancel()
            
        if timeout_triggered[0]:
            raise TimeoutError(f"Timeout ved venting på svar fra {tool_name} etter {self.timeout} sekunder")
            
        if response is None:
            raise TimeoutError(f"Fikk ikke svar fra MCP-server etter {self.timeout} sekunder")
            
        return response
    
    def test_semantic_search(self):
        """Test semantic_search-verktøyet."""
        logger.info("Tester semantic_search...")
        
        try:
            # Send en søkeforespørsel
            query = "Hva er definisjonen av offentlig dokument?"
            top_k = 2
            
            response = self.call_tool("semantic_search", {
                "query": query,
                "top_k": top_k
            })
            
            # Evaluer responsen
            if isinstance(response, list) and len(response) > 0:
                self.results["semantic_search"] = {
                    "success": True,
                    "duration": time.time() - self.results.get("start_time", 0),
                    "result_count": len(response),
                    "first_result": response[0] if response else None
                }
                
                logger.info(f"semantic_search: Suksess, returnerte {len(response)} resultater")
                
                # Returner første dokumentets ID for neste test
                if response and "id" in response[0]:
                    return response[0]["id"]
            else:
                self.results["semantic_search"] = {
                    "success": False,
                    "duration": time.time() - self.results.get("start_time", 0),
                    "error": "Ingen resultater returnert"
                }
                logger.error("semantic_search: Ingen resultater returnert")
                
            return None
            
        except Exception as e:
            self.results["semantic_search"] = {
                "success": False,
                "duration": time.time() - self.results.get("start_time", 0),
                "error": str(e)
            }
            logger.error(f"semantic_search: Feil: {str(e)}")
            return None
    
    def test_get_document(self, doc_id):
        """Test get_document-verktøyet."""
        if not doc_id:
            logger.warning("get_document: Ingen dokument-ID tilgjengelig for testing")
            self.results["get_document"] = {
                "success": False,
                "duration": 0,
                "error": "Ingen dokument-ID tilgjengelig"
            }
            return
            
        logger.info(f"Tester get_document med ID: {doc_id}...")
        
        try:
            # Send en get_document-forespørsel
            response = self.call_tool("get_document", {
                "id": doc_id
            })
            
            # Evaluer responsen
            if response:
                self.results["get_document"] = {
                    "success": True,
                    "duration": time.time() - self.results.get("start_time", 0),
                    "content_length": len(response) if isinstance(response, str) else len(json.dumps(response)),
                    "sample": response[:100] + "..." if isinstance(response, str) and len(response) > 100 else response
                }
                logger.info("get_document: Suksess, dokument mottatt")
            else:
                self.results["get_document"] = {
                    "success": False,
                    "duration": time.time() - self.results.get("start_time", 0),
                    "error": "Tomt svar"
                }
                logger.error("get_document: Tomt svar mottatt")
                
        except Exception as e:
            self.results["get_document"] = {
                "success": False,
                "duration": time.time() - self.results.get("start_time", 0),
                "error": str(e)
            }
            logger.error(f"get_document: Feil: {str(e)}")
    
    def test_direct_communication(self):
        """Test direkte kommunikasjon med MCP-serveren."""
        logger.info("Tester direkte kommunikasjon med MCP-serveren...")
        
        if self.process is None:
            logger.error("MCP-server kjører ikke")
            return False
        
        try:
            # En enkel echo-kommando for å teste kommunikasjon
            simple_request = {
                "name": "echo",
                "params": {
                    "message": "Hello, MCP!"
                }
            }
            
            # Send forespørselen
            request_json = json.dumps(simple_request)
            logger.info(f"Sender enkel echo-forespørsel: {request_json}")
            self.process.stdin.write(request_json + "\n")
            self.process.stdin.flush()
            
            # Les stderr for å sjekke om det er noen feilmeldinger
            stderr_output = ""
            while self.process.poll() is None:
                stderr_line = self.process.stderr.readline().strip()
                if stderr_line:
                    stderr_output += stderr_line + "\n"
                    logger.warning(f"MCP stderr: {stderr_line}")
                else:
                    break  # Ingen flere linjer å lese
            
            if stderr_output:
                logger.warning(f"MCP-serveren rapporterer feil:\n{stderr_output}")
                # Men fortsett testen, siden serveren fortsatt kan fungere
            
            # Sjekk om prosessen fortsatt kjører
            if self.process.poll() is not None:
                logger.error(f"MCP-server avsluttet med kode {self.process.returncode}")
                return False
                
            logger.info("Grunnleggende kommunikasjonstest bestått")
            
            # Test at semantic_search er implementert
            logger.info("Sjekker om semantic_search er implementert...")
            simple_search = {
                "name": "semantic_search",
                "params": {
                    "query": "test",
                    "top_k": 1
                }
            }
            
            # Send forespørselen uten å vente på svar
            request_json = json.dumps(simple_search)
            logger.info(f"Sender test-søkeforespørsel: {request_json}")
            self.process.stdin.write(request_json + "\n")
            self.process.stdin.flush()
            
            # Gi litt tid for å se om serveren krasjer
            time.sleep(1)
            
            # Sjekk om prosessen fortsatt kjører
            if self.process.poll() is not None:
                logger.error(f"MCP-server krasjet ved semantic_search med kode {self.process.returncode}")
                stderr = self.process.stderr.read()
                logger.error(f"stderr: {stderr}")
                return False
            
            logger.info("MCP-server håndterte semantic_search-kallet uten å krasje")
            self.results["direct_communication"] = {
                "success": True,
                "message": "MCP-server responderte korrekt på grunnleggende kommunikasjonstest"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Feil ved direkte kommunikasjonstest: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.results["direct_communication"] = {
                "success": False,
                "message": f"Feil ved direkte kommunikasjonstest: {str(e)}"
            }
            
            return False
    
    def run_tests(self):
        """Kjør alle tester."""
        self.results["start_time"] = time.time()
        
        try:
            # Start serveren
            if not self.start_server():
                self.results["overall"] = {
                    "success": False,
                    "message": "Kunne ikke starte MCP-serveren"
                }
                return self.results
            
            # Test direkte kommunikasjon med serveren først
            if not self.test_direct_communication():
                logger.error("Direkte kommunikasjonstest feilet. Avbryter ytterligere tester.")
                self.results["overall"] = {
                    "success": False,
                    "message": "Kunne ikke etablere grunnleggende kommunikasjon med MCP-serveren"
                }
                return self.results
            
            # Test semantic_search
            doc_id = self.test_semantic_search()
            
            # Test get_document hvis vi fikk en dokument-ID
            if doc_id:
                self.test_get_document(doc_id)
            
            # Vurder samlet resultat
            semantic_search_success = self.results.get("semantic_search", {}).get("success", False)
            get_document_success = self.results.get("get_document", {}).get("success", False)
            
            self.results["overall"] = {
                "success": semantic_search_success and (get_document_success or not doc_id),
                "message": "Alle tester bestått" if semantic_search_success and (get_document_success or not doc_id) else "En eller flere tester feilet",
                "duration": time.time() - self.results["start_time"]
            }
            
        except Exception as e:
            logger.error(f"Feil ved kjøring av tester: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.results["overall"] = {
                "success": False,
                "message": f"Feil ved kjøring av tester: {str(e)}",
                "duration": time.time() - self.results["start_time"]
            }
            
        finally:
            # Stopp serveren
            self.stop_server()
            
        return self.results

def main():
    """Hovedfunksjon."""
    # Last inn miljøvariabler
    load_env_vars()
    
    # Opprett og kjør testen
    test = MCPServerTest(debug=True)
    results = test.run_tests()
    
    # Skriv ut resultater
    print("\n" + "="*80)
    print("MCP-SERVER TESTRESULTATER")
    print("="*80)
    
    overall_success = results.get("overall", {}).get("success", False)
    overall_message = results.get("overall", {}).get("message", "Ukjent resultat")
    overall_duration = results.get("overall", {}).get("duration", 0)
    
    print(f"Samlet resultat: {'✅ BESTÅTT' if overall_success else '❌ FEILET'}")
    print(f"Melding: {overall_message}")
    print(f"Varighet: {overall_duration:.2f} sekunder")
    print("-"*80)
    
    # Detaljerte resultater
    for test_name, test_result in results.items():
        if test_name in ["start_time", "overall"]:
            continue
            
        success = test_result.get("success", False)
        print(f"{test_name}: {'✅ BESTÅTT' if success else '❌ FEILET'}")
        
        for key, value in test_result.items():
            if key != "success":
                print(f"  - {key}: {value}")
    
    print("="*80)
    
    # Skriv resultater til JSON-fil
    results_file = os.path.join(PROJECT_ROOT, "mcp-test-results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Testresultater skrevet til {results_file}")
    
    # Returner exit-kode
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main()) 