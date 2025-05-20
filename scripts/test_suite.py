#!/usr/bin/env python3
"""
Test-suite for Lovdata RAG-agent.

Denne test-suiten tester alle nødvendige aspekter av RAG-agenten for norske lover:
1. Miljøvariabler og konfigurasjon
2. Pinecone-tilkobling 
3. OpenAI embedding-generering
4. Dokumentinnhenting
5. MCP-server og kommunikasjon
6. Komplette integrasjonstester

Bruk:
    python test_suite.py --all         # Kjør alle tester
    python test_suite.py --env         # Test miljøvariabler
    python test_suite.py --pinecone    # Test Pinecone-tilkobling
    python test_suite.py --embeddings  # Test embedding-generering
    python test_suite.py --retrieval   # Test dokumentinnhenting
    python test_suite.py --mcp         # Test MCP-server
    python test_suite.py --integration # Test komplett integrasjon
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Legg til prosjektets rotmappe i sys.path for å støtte importer
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Skriv til stderr, ikke stdout som kan forstyrre MCP-kommunikasjon
)
logger = logging.getLogger("test-suite")

# Konfigurasjon for MCP-server
MCP_SERVER_SCRIPT = os.path.join(project_root, "src", "mcp_server.py")

class TestResult:
    """Klasse for å representere testresultater."""
    
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.message = ""
        self.details = {}
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
    
    def complete(self, success: bool, message: str, details: Dict[str, Any] = None):
        """Marker testen som fullført."""
        self.success = success
        self.message = message
        self.details = details or {}
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Konverter testresultatet til en dict."""
        return {
            "name": self.name,
            "success": self.success,
            "message": self.message,
            "details": self.details,
            "duration": self.duration
        }
    
    def __str__(self) -> str:
        """Streng-representasjon av testresultatet."""
        status = "✅ BESTÅTT" if self.success else "❌ FEILET"
        duration = f"{self.duration:.2f}s" if self.duration is not None else "N/A"
        return f"{status} {self.name} ({duration}): {self.message}"

class TestSuite:
    """Hovedklasse for test-suiten."""
    
    def __init__(self, args):
        """Initialiser test-suiten."""
        self.args = args
        self.results = []
        self.load_env()
    
    def load_env(self):
        """Last inn miljøvariabler fra .env-fil."""
        dotenv_path = os.path.join(project_root, ".env")
        if os.path.exists(dotenv_path):
            logger.info(f"Laster miljøvariabler fra {dotenv_path}")
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=dotenv_path)
                logger.info("Miljøvariabler lastet")
            except ImportError:
                logger.warning("python-dotenv ikke installert, miljøvariabler lastes ikke fra .env-fil")
        else:
            logger.warning(f"Ingen .env-fil funnet på {dotenv_path}")
    
    def run_tests(self):
        """Kjør de valgte testene."""
        if self.args.all or self.args.env:
            self.test_environment()
        
        if self.args.all or self.args.pinecone:
            self.test_pinecone_connection()
        
        if self.args.all or self.args.embeddings:
            self.test_embedding_generation()
        
        if self.args.all or self.args.retrieval:
            self.test_document_retrieval()
        
        if self.args.all or self.args.mcp:
            self.test_mcp_server()
        
        if self.args.all or self.args.integration:
            self.test_integration()
        
        # Skriv ut oppsummering
        self.print_summary()
    
    def add_result(self, result: TestResult):
        """Legg til et testresultat i listen."""
        self.results.append(result)
        logger.info(str(result))
    
    def print_summary(self):
        """Skriv ut en oppsummering av testresultatene."""
        print("\n" + "="*80)
        print("TESTRESULTATER")
        print("="*80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        
        for result in self.results:
            print(str(result))
        
        print("\n" + "-"*80)
        print(f"Totalt: {total} tester")
        print(f"Bestått: {passed} tester")
        print(f"Feilet: {failed} tester")
        print("="*80)
        
        # Skriv resultater til JSON-fil
        results_file = os.path.join(project_root, "test-results.json")
        with open(results_file, "w") as f:
            json.dump({
                "total": total,
                "passed": passed,
                "failed": failed,
                "tests": [r.to_dict() for r in self.results]
            }, f, indent=2)
        
        logger.info(f"Testresultater skrevet til {results_file}")
    
    def test_environment(self):
        """Test at miljøvariablene er riktig konfigurert."""
        result = TestResult("Miljøvariabler")
        
        try:
            # Importer config-modulen
            from config import validate_config, get_config_dict
            
            # Sjekk at konfigurasjonen er gyldig
            config_valid, config_errors = validate_config()
            config_dict = get_config_dict()
            
            # Sjekk nødvendige miljøvariabler
            env_vars = {
                "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY"),
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
                "PINECONE_INDEX_NAME": os.environ.get("PINECONE_INDEX_NAME"),
            }
            
            missing_vars = [k for k, v in env_vars.items() if v is None]
            
            if config_valid and not missing_vars:
                result.complete(
                    True, 
                    "Alle nødvendige miljøvariabler er satt",
                    {"config": config_dict}
                )
            else:
                result.complete(
                    False, 
                    f"Manglende miljøvariabler: {', '.join(missing_vars)}" if missing_vars else f"Konfigurasjonsvalidering feilet: {', '.join(config_errors)}",
                    {
                        "missing_vars": missing_vars,
                        "config_errors": config_errors,
                        "config": config_dict
                    }
                )
        except Exception as e:
            result.complete(
                False, 
                f"Feil ved testing av miljøvariabler: {str(e)}",
                {"error": str(e)}
            )
        
        self.add_result(result)
    
    def test_pinecone_connection(self):
        """Test tilkobling til Pinecone."""
        result = TestResult("Pinecone-tilkobling")
        
        try:
            import pinecone
            
            # Hent API-nøkkel og indeksnavn
            api_key = os.environ.get("PINECONE_API_KEY")
            index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-index")
            host = os.environ.get("PINECONE_HOST")
            
            if not api_key:
                result.complete(
                    False, 
                    "PINECONE_API_KEY er ikke satt",
                    {"api_key": None}
                )
                self.add_result(result)
                return
            
            # Initialiser Pinecone
            pc = pinecone.Pinecone(api_key=api_key)
            
            # Test tilkobling til indeks
            if host:
                # Bruk host direkte
                index = pc.Index(host=host)
                stats = index.describe_index_stats()
                
                result.complete(
                    True, 
                    f"Vellykket tilkobling til Pinecone-indeks via host",
                    {"host": host, "stats": str(stats)}
                )
            else:
                # List indekser og finn indeksen vår
                indexes = pc.list_indexes()
                index_names = [idx.name for idx in indexes]
                
                if index_name in index_names:
                    index_info = pc.describe_index(name=index_name)
                    index = pc.Index(host=index_info.host)
                    stats = index.describe_index_stats()
                    
                    result.complete(
                        True, 
                        f"Vellykket tilkobling til Pinecone-indeks {index_name}",
                        {"index": index_name, "host": index_info.host, "stats": str(stats)}
                    )
                else:
                    result.complete(
                        False, 
                        f"Indeks {index_name} ikke funnet i tilgjengelige indekser: {index_names}",
                        {"index": index_name, "available_indexes": index_names}
                    )
        except Exception as e:
            result.complete(
                False, 
                f"Feil ved tilkobling til Pinecone: {str(e)}",
                {"error": str(e)}
            )
        
        self.add_result(result)
    
    def test_embedding_generation(self):
        """Test generering av embeddings med OpenAI."""
        result = TestResult("Embedding-generering")
        
        try:
            from langchain_openai import OpenAIEmbeddings
            
            # Hent API-nøkkel
            api_key = os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                result.complete(
                    False, 
                    "OPENAI_API_KEY er ikke satt",
                    {"api_key": None}
                )
                self.add_result(result)
                return
            
            # Generer embedding
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            test_text = "Dette er en test av embedding-generering"
            embedding_vector = embeddings.embed_query(test_text)
            
            result.complete(
                True, 
                f"Vellykket generering av embedding med {len(embedding_vector)} dimensjoner",
                {"dimensions": len(embedding_vector), "sample": embedding_vector[:5]}
            )
        except Exception as e:
            result.complete(
                False, 
                f"Feil ved generering av embedding: {str(e)}",
                {"error": str(e)}
            )
        
        self.add_result(result)
    
    def test_document_retrieval(self):
        """Test henting av dokumenter fra vektor-databasen."""
        result = TestResult("Dokumenthenting")
        
        try:
            import asyncio
            from shared import retrieval
            
            # Test søk etter dokumenter
            async def run_search():
                query = "offentlighetsloven innsyn"
                docs = await retrieval.search_documents(query, limit=3)
                return docs
            
            # Kjør asyncio-funksjonen
            loop = asyncio.get_event_loop()
            docs = loop.run_until_complete(run_search())
            
            if docs and len(docs) > 0:
                result.complete(
                    True, 
                    f"Vellykket henting av {len(docs)} dokumenter",
                    {
                        "count": len(docs),
                        "sample": {
                            "metadata": docs[0].metadata if hasattr(docs[0], "metadata") else None,
                            "content_preview": docs[0].page_content[:100] if hasattr(docs[0], "page_content") else None
                        }
                    }
                )
            else:
                result.complete(
                    False, 
                    "Ingen dokumenter funnet ved søk",
                    {"docs": None}
                )
        except Exception as e:
            result.complete(
                False, 
                f"Feil ved henting av dokumenter: {str(e)}",
                {"error": str(e)}
            )
        
        self.add_result(result)
    
    def test_mcp_server(self):
        """Test at MCP-serveren starter og svarer på kommandoer."""
        result = TestResult("MCP-server")
        
        # Opprett en MCP-server-prosess
        process = None
        try:
            # Start MCP-serveren
            process = subprocess.Popen(
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
            if process.poll() is not None:
                stderr = process.stderr.read()
                result.complete(
                    False, 
                    f"MCP-server avsluttet med kode {process.returncode}",
                    {"returncode": process.returncode, "stderr": stderr}
                )
                self.add_result(result)
                return
            
            # Test en enkel kommando (ping)
            request = {
                "name": "semantic_search",
                "params": {"query": "test", "top_k": 1}
            }
            
            # Send kommandoen
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            # Les responsen med timeout
            response = None
            timeout = time.time() + 30  # 30 sekunder timeout
            
            while response is None and time.time() < timeout:
                line = process.stdout.readline().strip()
                if not line:
                    time.sleep(0.1)
                    continue
                
                try:
                    response = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
            
            if response is not None:
                result.complete(
                    True, 
                    "MCP-server svarte på kommando",
                    {"response": response}
                )
            else:
                result.complete(
                    False, 
                    "Timeout ved venting på svar fra MCP-server",
                    {"timeout": 30}
                )
        except Exception as e:
            result.complete(
                False, 
                f"Feil ved testing av MCP-server: {str(e)}",
                {"error": str(e)}
            )
        finally:
            # Avslutt serveren
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        self.add_result(result)
    
    def test_integration(self):
        """Kjør en fullstendig integrasjonstest."""
        result = TestResult("Integrasjonstest")
        
        # Opprett en MCP-server-prosess
        process = None
        try:
            # Start MCP-serveren
            process = subprocess.Popen(
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
            if process.poll() is not None:
                stderr = process.stderr.read()
                result.complete(
                    False, 
                    f"MCP-server avsluttet med kode {process.returncode}",
                    {"returncode": process.returncode, "stderr": stderr}
                )
                self.add_result(result)
                return
            
            # Test 1: Søk etter dokumenter
            query = "Hva sier offentlighetsloven om innsyn i offentlige dokumenter?"
            search_request = {
                "name": "semantic_search",
                "params": {"query": query, "top_k": 3}
            }
            
            # Send kommandoen
            process.stdin.write(json.dumps(search_request) + "\n")
            process.stdin.flush()
            
            # Les responsen med timeout
            search_response = None
            timeout = time.time() + 30  # 30 sekunder timeout
            
            while search_response is None and time.time() < timeout:
                line = process.stdout.readline().strip()
                if not line:
                    time.sleep(0.1)
                    continue
                
                try:
                    search_response = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
            
            if search_response is None or not isinstance(search_response, list) or len(search_response) == 0:
                result.complete(
                    False, 
                    "Ingen søkeresultater mottatt fra MCP-server",
                    {"response": search_response}
                )
                self.add_result(result)
                return
            
            # Test 2: Hent dokument
            doc_id = search_response[0].get("id")
            if not doc_id:
                result.complete(
                    False, 
                    "Søkeresultatet inneholder ikke dokument-ID",
                    {"search_response": search_response}
                )
                self.add_result(result)
                return
            
            # Send forespørsel om dokument
            doc_request = {
                "name": "get_document",
                "params": {"id": doc_id}
            }
            
            process.stdin.write(json.dumps(doc_request) + "\n")
            process.stdin.flush()
            
            # Les responsen med timeout
            doc_response = None
            timeout = time.time() + 30  # 30 sekunder timeout
            
            while doc_response is None and time.time() < timeout:
                line = process.stdout.readline().strip()
                if not line:
                    time.sleep(0.1)
                    continue
                
                try:
                    doc_response = json.loads(line)
                    break
                except json.JSONDecodeError:
                    try:
                        # Prøv å tolke som ren tekst hvis JSON-parsing feiler
                        doc_response = line
                        break
                    except:
                        pass
            
            if doc_response:
                result.complete(
                    True, 
                    "Fullstendig integrasjonstest vellykket",
                    {
                        "search_results": search_response,
                        "document": doc_response[:100] + "..." if isinstance(doc_response, str) and len(doc_response) > 100 else doc_response
                    }
                )
            else:
                result.complete(
                    False, 
                    "Timeout ved venting på dokument fra MCP-server",
                    {"search_results": search_response, "document": None}
                )
        except Exception as e:
            result.complete(
                False, 
                f"Feil ved integrasjonstest: {str(e)}",
                {"error": str(e)}
            )
        finally:
            # Avslutt serveren
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        self.add_result(result)

def main():
    """Hovedfunksjon."""
    parser = argparse.ArgumentParser(description="Test-suite for Lovdata RAG-agent")
    
    # Legg til argumenter
    parser.add_argument("--all", action="store_true", help="Kjør alle tester")
    parser.add_argument("--env", action="store_true", help="Test miljøvariabler")
    parser.add_argument("--pinecone", action="store_true", help="Test Pinecone-tilkobling")
    parser.add_argument("--embeddings", action="store_true", help="Test embedding-generering")
    parser.add_argument("--retrieval", action="store_true", help="Test dokumenthenting")
    parser.add_argument("--mcp", action="store_true", help="Test MCP-server")
    parser.add_argument("--integration", action="store_true", help="Test komplett integrasjon")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detaljert loggføring")
    
    args = parser.parse_args()
    
    # Hvis ingen tester er valgt, velg alle
    if not (args.all or args.env or args.pinecone or args.embeddings or args.retrieval or args.mcp or args.integration):
        args.all = True
    
    # Sett loggnivå
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Kjør test-suite
    suite = TestSuite(args)
    suite.run_tests()

if __name__ == "__main__":
    main() 