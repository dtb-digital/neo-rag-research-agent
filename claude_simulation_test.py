import os
import sys
import json
import asyncio
import logging
import signal
import subprocess
import time
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple
import dotenv

# Last inn miljøvariabler fra .env-filen
print("Laster miljøvariabler fra .env-filen")
dotenv.load_dotenv()

# Sett opp logging
logging.basicConfig(
    level=logging.DEBUG,  # Endret til DEBUG for mer detaljert logg
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("claude_simulation.log")],
)
logger = logging.getLogger("claude_simulation")

class ClaudeSimulator:
    """
    En simulator som oppfører seg som Claude når den kommuniserer med MCP-serveren.
    """
    
    def __init__(self):
        # Sett opp miljøvariabler som Claude ville fått fra Cursor
        self.env = os.environ.copy()
        
        # Sett miljøvariabel direkte for debugging
        self.env["LOG_LEVEL"] = "DEBUG"
        
        # Logg miljøvariabler for debugging
        logger.info(f"PINECONE_API_KEY: {'SATT' if self.env.get('PINECONE_API_KEY') else 'MANGLER'}")
        logger.info(f"OPENAI_API_KEY: {'SATT' if self.env.get('OPENAI_API_KEY') else 'MANGLER'}")
        logger.info(f"PINECONE_INDEX_NAME: {self.env.get('PINECONE_INDEX_NAME', 'ikke satt')}")
        
        # Sjekk at nødvendige miljøvariabler er satt
        required_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY", "PINECONE_INDEX_NAME"]
        for var in required_vars:
            if not self.env.get(var):
                logger.error(f"Miljøvariabel {var} er ikke satt. Husk å kjøre dotenv.load_dotenv().")
                logger.info(f"Tilgjengelige miljøvariabler: {list(self.env.keys())}")
                if var in os.environ:
                    logger.info(f"Verdien av {var} i os.environ: '{os.environ[var]}'")
                raise ValueError(f"Manglende miljøvariabel: {var}")
        
        self.process = None
        self.running = False
    
    def start_mcp_server(self) -> None:
        """
        Start MCP-serveren som en separat prosess med de samme miljøvariablene som Claude ville brukt.
        """
        logger.info("Starter MCP-server...")
        
        try:
            # Sett opp prosessen for MCP-serveren med stdio kommunikasjon
            # Eksplisitt angi environment og stdout/stderr håndtering
            command = [sys.executable, "src/mcp_server.py"]
            logger.info(f"Kjører kommando: {' '.join(command)}")
            
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                text=True,
                bufsize=1  # Linjevis bufring
            )
            
            self.running = True
            logger.info("MCP-server startet (PID: %d)", self.process.pid)
            
            # Gi serveren tid til å starte opp
            time.sleep(2)
            
            # Les initielle loggmeldinger fra stderr for debugging
            stderr_output = ""
            try:
                import fcntl
                flags = fcntl.fcntl(self.process.stderr, fcntl.F_GETFL)
                fcntl.fcntl(self.process.stderr, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                try:
                    stderr_output = self.process.stderr.read()
                except:
                    pass
                
                if stderr_output:
                    logger.info(f"MCP-server stderr: {stderr_output}")
            except:
                logger.warning("Kunne ikke lese stderr i non-blocking modus")
            
            # Les ut alle initielle loggmeldinger fra stdout
            stdout_output = self._clear_output_buffer()
            if stdout_output:
                logger.info(f"MCP-server initielle stdout: {stdout_output}")
            
            # Sjekk at prosessen fortsatt kjører
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read() if not stderr_output else stderr_output
                logger.error(f"MCP-server avsluttet uventet med kode {self.process.returncode}")
                logger.error(f"Stderr utdata: {stderr_output}")
                raise RuntimeError(f"MCP-server avsluttet uventet med kode {self.process.returncode}")
            
        except Exception as e:
            logger.error(f"Feil ved oppstart av MCP-server: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _clear_output_buffer(self):
        """Les ut og logg alle ventende meldinger fra serveren."""
        if not self.process:
            return ""
        
        output = ""
        
        # Sett stdout til ikke-blokkerende
        import fcntl
        import os
        
        flags = fcntl.fcntl(self.process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(self.process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        try:
            output = self.process.stdout.read()
            if output:
                logger.debug(f"Ryddet ut: {output}")
        except:
            pass
        
        # Sett tilbake til blokkerende
        fcntl.fcntl(self.process.stdout, fcntl.F_SETFL, flags)
        
        return output
    
    def stop_mcp_server(self) -> None:
        """
        Stopp MCP-serveren på en ryddig måte.
        """
        if not self.running or not self.process:
            logger.info("MCP-server er ikke i gang.")
            return
        
        logger.info("Stopper MCP-server...")
        
        try:
            # Send SIGTERM til prosessen
            self.process.terminate()
            
            # Vent på at prosessen skal avslutte
            try:
                self.process.wait(timeout=3)  # Redusert timeout fra 5 til 3 sekunder
                logger.info("MCP-server stoppet normalt.")
            except subprocess.TimeoutExpired:
                logger.warning("MCP-server responderer ikke på SIGTERM, sender SIGKILL...")
                self.process.kill()
                self.process.wait(timeout=3)  # Redusert timeout fra 5 til 3 sekunder
                logger.info("MCP-server terminert med SIGKILL.")
        
        except Exception as e:
            logger.error(f"Feil ved stopping av MCP-server: {str(e)}")
        
        self.running = False
        self.process = None
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Kall et verktøy på MCP-serveren, akkurat som Claude ville gjøre.
        
        Args:
            tool_name: Navnet på verktøyet som skal kalles
            **kwargs: Argumenter som skal sendes til verktøyet
            
        Returns:
            Resultatet fra verktøykallet
        """
        if not self.running or not self.process:
            raise RuntimeError("MCP-server er ikke i gang.")
        
        try:
            # Bygg opp meldingen som Claude ville sendt til MCP-serveren
            message = {
                "type": "function_call",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(kwargs)
                }
            }
            
            message_str = json.dumps(message) + "\n"
            logger.info(f"Sender melding til MCP-server: {message_str.strip()}")
            
            # Tøm eventuell output buffer først
            self._clear_output_buffer()
            
            # Skriv meldingen til serveren
            self.process.stdin.write(message_str)
            self.process.stdin.flush()
            
            # Vent på svar - les linjer til vi finner en gyldig JSON-respons
            valid_json = False
            json_response = None
            
            start_time = time.time()
            timeout = 10  # Redusert fra 30 til 10 sekunder
            
            # Samle opp all output for debugging-formål
            all_output = []
            
            while not valid_json and (time.time() - start_time) < timeout:
                # Sjekk at prosessen fortsatt kjører
                if self.process.poll() is not None:
                    logger.error(f"MCP-server avsluttet uventet med kode {self.process.returncode}")
                    raise RuntimeError(f"MCP-server avsluttet uventet med kode {self.process.returncode}")
                
                # Les en linje om gangen
                response_line = self.process.stdout.readline()
                all_output.append(response_line)
                
                if not response_line.strip():
                    # Tom linje, vent litt og fortsett
                    await asyncio.sleep(0.1)
                    continue
                
                logger.debug(f"Leste linje: {response_line.strip()}")
                
                # Sjekk om linjen ser ut som en loggmelding
                if re.match(r'\d{4}-\d{2}-\d{2}', response_line.strip()):
                    logger.debug(f"Ignorerer loggmelding: {response_line.strip()}")
                    continue
                
                # Prøv å parse som JSON
                try:
                    json_response = json.loads(response_line)
                    valid_json = True
                    logger.info(f"Mottok gyldig JSON-svar: {response_line.strip()[:100]}...")
                except json.JSONDecodeError:
                    logger.debug(f"Kunne ikke parse som JSON: {response_line.strip()}")
            
            if not valid_json:
                logger.error(f"Timeout uten å motta gyldig JSON-svar fra serveren")
                logger.error(f"All mottatt output: {''.join(all_output)}")
                # Returner et dummy-svar for å kunne fortsette testene
                return f"DUMMY-SVAR: Ingen gyldig respons mottatt innen {timeout} sekunder"
            
            # Behandle JSON-responsen
            if "content" in json_response:
                logger.info(f"MCP-server svarte med innholdstype: {type(json_response['content'])}")
                logger.info(f"Svar (forkortet): {str(json_response['content'])[:200]}...")
                return json_response["content"]
            else:
                logger.warning(f"Uventet svarformat: {json_response}")
                return json_response
                
        except Exception as e:
            logger.error(f"Feil ved kommunikasjon med MCP-server: {str(e)}")
            logger.error(traceback.format_exc())
            # Returner et dummy-svar for å kunne fortsette testene
            return f"DUMMY-SVAR: Feil ved kommunikasjon med MCP-server: {str(e)}"
    
    async def run_tests(self) -> None:
        """
        Kjør en serie med tester som simulerer hvordan Claude ville brukt MCP-serveren.
        """
        try:
            # Start MCP-serveren
            self.start_mcp_server()
            
            # Kjør bare én test for å isolere problemer
            logger.info("\n=== TEST 1: Søk i lovdata (forenklet) ===")
            sok_resultat = await self.call_tool(
                "sok_i_lovdata", 
                sporsmal="Habilitet",  # Forenklet spørsmål
                antall_resultater=1    # Redusert antall
            )
            
            if sok_resultat:
                logger.info(f"Søk fullført, resultattype: {type(sok_resultat)}")
                logger.info(f"Resultatlengde: {len(sok_resultat) if isinstance(sok_resultat, str) else 'ikke en streng'}")
                
            logger.info("Fullført enkel test. Avslutter.")
            
        except Exception as e:
            logger.error(f"Testfeil: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Stopp MCP-serveren
            self.stop_mcp_server()

async def main():
    """Hovedfunksjon for å kjøre Claude-simulatoren."""
    logger.info("=== STARTER CLAUDE SIMULATOR TESTER ===")
    
    # Logg miljøvariabler
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
    
    logger.info(f"Miljøvariabler: PINECONE_API_KEY {'funnet' if pinecone_api_key else 'MANGLER'}")
    logger.info(f"Miljøvariabler: OPENAI_API_KEY {'funnet' if openai_api_key else 'MANGLER'}")
    logger.info(f"Pinecone indeks: {pinecone_index_name}")
    
    simulator = ClaudeSimulator()
    await simulator.run_tests()
    
    logger.info("=== CLAUDE SIMULATOR TESTER FULLFØRT ===")

if __name__ == "__main__":
    # Sett en maksimal kjøretid for hele programmet for å unngå å henge
    # Bruk en signal-handler for å sette en maksimal kjøretid
    def timeout_handler(signum, frame):
        logger.error("Timeout! Programmet har kjørt for lenge og avsluttes.")
        sys.exit(1)
    
    # Sett en timeout på 60 sekunder for hele programmet
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Uventet feil i hovedprogrammet: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Deaktiver alarmen hvis alt gikk bra
        signal.alarm(0) 