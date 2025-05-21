"""
MCP-server for Lovdata RAG-agent.

Implementerer en Model Context Protocol (MCP) server som bruker stdio-transport
for å kommunisere med Claude og eksponere verktøy for å søke i lovdata.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional
import dotenv
from pathlib import Path
import logging

# Sett opp en enkel initiell logger til stderr før vi laster full logging
init_logger = logging.getLogger("init")
init_logger.setLevel(logging.INFO)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
init_logger.addHandler(stderr_handler)

# Last inn miljøvariabler fra .env-filen
init_logger.info("Laster miljøvariabler fra .env-filen...")
dotenv.load_dotenv()

# Legg til prosjektets rotmappe i sys.path for å støtte importer
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
# Legg til src-mappen i sys.path for å støtte begge importstiler
sys.path.append(os.path.join(project_root, "src"))

# Importer logger.py først for å sette opp grunnleggende logging
from shared.logging_config import logger, setup_logger, configure_logging

# Konfigurer logging eksplisitt for å sikre at alt er satt opp korrekt
configure_logging()

# Nå er logging konfigurert, så vi kan fortsette med resten av importen
init_logger.info("Logging konfigurasjon initialisert")

# Hvis kritiske miljøvariabler mangler, prøv å lese direkte fra .env-filen
env_file = Path('.env')
if env_file.exists() and (
    not os.environ.get("PINECONE_API_KEY") or 
    not os.environ.get("OPENAI_API_KEY") or
    not os.environ.get("LANGSMITH_API_KEY")
):
    init_logger.info("Noen miljøvariabler mangler, prøver å lese direkte fra .env-filen...")
    try:
        env_content = env_file.read_text()
        
        # Hent nøkler med regex
        import re
        pinecone_match = re.search(r'PINECONE_API_KEY=(.+)', env_content)
        openai_match = re.search(r'OPENAI_API_KEY=(.+)', env_content)
        langsmith_api_match = re.search(r'LANGSMITH_API_KEY=(.+)', env_content)
        langsmith_tracing_match = re.search(r'LANGSMITH_TRACING=(.+)', env_content)
        langsmith_project_match = re.search(r'LANGSMITH_PROJECT=(.+)', env_content)
        langsmith_endpoint_match = re.search(r'LANGSMITH_ENDPOINT=(.+)', env_content)
        
        # Sett miljøvariablene direkte
        if pinecone_match and not os.environ.get("PINECONE_API_KEY"):
            os.environ["PINECONE_API_KEY"] = pinecone_match.group(1).strip()
            init_logger.info("Satt PINECONE_API_KEY direkte fra .env")
        
        if openai_match and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_match.group(1).strip()
            init_logger.info("Satt OPENAI_API_KEY direkte fra .env")
            
        # Legg til LANGSMITH-miljøvariablene
        if langsmith_api_match and not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_API_KEY direkte fra .env")
            
        if langsmith_tracing_match and not os.environ.get("LANGSMITH_TRACING"):
            os.environ["LANGSMITH_TRACING"] = langsmith_tracing_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_TRACING direkte fra .env")
            
        if langsmith_project_match and not os.environ.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = langsmith_project_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_PROJECT direkte fra .env")
            
        if langsmith_endpoint_match and not os.environ.get("LANGSMITH_ENDPOINT"):
            os.environ["LANGSMITH_ENDPOINT"] = langsmith_endpoint_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_ENDPOINT direkte fra .env")
    except Exception as e:
        init_logger.error(f"Kunne ikke lese direkte fra .env: {str(e)}")

# Sett standard verdi for Pinecone-indeksen
if not os.environ.get("PINECONE_INDEX_NAME"):
    os.environ["PINECONE_INDEX_NAME"] = "lovdata-paragraf-test"
    init_logger.info("Satt standard PINECONE_INDEX_NAME til 'lovdata-paragraf-test'")

# Sett standard verdi for LANGSMITH_TRACING hvis den ikke er satt
if not os.environ.get("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"
    init_logger.info("Satt standard LANGSMITH_TRACING til 'true'")

# Sett miljøvariabelen LOG_LEVEL til INFO direkte før import av FastMCP
os.environ["LOG_LEVEL"] = "INFO"

# Fortell FastMCP å logge til stderr i stedet for stdout
os.environ["FASTMCP_LOG_TO_STDERR"] = "true"

from fastmcp import FastMCP
from src.utils import truncate_text

# Import retrieval_graph og andre nødvendige moduler
try:
    # Forsøk å importere retrieval_graph for søkefunksjonalitet
    from retrieval_graph import graph as retrieval_graph
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import HumanMessage
except ImportError as e:
    # Stopp programmet hvis retrieval_graph ikke er tilgjengelig
    logger.error(f"Kunne ikke importere retrieval_graph: {str(e)}")
    logger.error("Kan ikke starte MCP-serveren uten retrieval_graph. Avslutter.")
    sys.exit(1)

# Opprett en spesifikk logger for MCP-serveren
mcp_logger = setup_logger("mcp-server")

class LovdataMCPServer:
    """MCP-server for Lovdata RAG-agent."""
    
    def __init__(self):
        """Initialiser MCP-serveren."""
        self.mcp = FastMCP("Lovdata RAG")
        
        # Logg miljøvariabel-status for debugging
        pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
        langsmith_api_key = os.environ.get("LANGSMITH_API_KEY", "")
        langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "")
        langsmith_project = os.environ.get("LANGSMITH_PROJECT", "")
        
        mcp_logger.info(f"Miljøvariabler: PINECONE_API_KEY {'funnet' if pinecone_api_key else 'MANGLER'}")
        mcp_logger.info(f"Miljøvariabler: OPENAI_API_KEY {'funnet' if openai_api_key else 'MANGLER'}")
        mcp_logger.info(f"Bruker Pinecone indeks: {pinecone_index_name}")
        mcp_logger.info(f"Miljøvariabler: LANGSMITH_API_KEY {'funnet' if langsmith_api_key else 'MANGLER'}")
        mcp_logger.info(f"Miljøvariabler: LANGSMITH_TRACING {'funnet' if langsmith_tracing else 'MANGLER'}")
        mcp_logger.info(f"Miljøvariabler: LANGSMITH_PROJECT {'funnet' if langsmith_project else 'MANGLER'}")
        
        # Sjekk at kritiske miljøvariabler er satt
        if not pinecone_api_key:
            mcp_logger.error("PINECONE_API_KEY er ikke satt! Vektorsøk vil ikke fungere")
        if not openai_api_key:
            mcp_logger.error("OPENAI_API_KEY er ikke satt! Embedding og LLM vil ikke fungere")
        if not langsmith_api_key and langsmith_tracing.lower() == "true":
            mcp_logger.error("LANGSMITH_API_KEY er ikke satt, men LANGSMITH_TRACING er aktivert! LangSmith-sporing vil ikke fungere")
        
        # Registrer verktøy
        self._register_tools()
        
        mcp_logger.info("MCP-server initialisert")
    
    def _register_tools(self):
        """Registrer verktøy for MCP-serveren."""
        
        @self.mcp.tool()
        async def sok_i_lovdata(sporsmal: str, antall_resultater: int = 10) -> str:
            """
            Søk etter relevante lovtekster, forskrifter og juridiske dokumenter basert på ditt spørsmål.
            
            BRUK DETTE VERKTØYET når du ønsker oppdatert informasjon om lovverk og forskrifter, hentet direkte fra lovdata.no.
            
            Dette verktøyet bruker en avansert juridisk modell til å forstå ditt spørsmål og returnerer 
            de mest relevante delene av norsk lovverk fra Lovdata. Verktøyet analyserer spørsmålet ditt,
            utfører semantisk søk i lovdatabasen, og formaterer resultatene slik at de er lett tilgjengelige.
            
            Args:
                sporsmal: Ditt juridiske spørsmål eller søkeord som du ønsker å finne relevante lover og forskrifter til
                antall_resultater: Antall resultater som skal returneres (standard: 10)
                
            Returns:
                Bearbeidet svar på spørsmålet, eller en liste med relevante lovtekster
                
            Eksempel på bruk:
                "Hva sier offentlighetsloven om innsyn i dokumenter?"
                "Hvilke regler gjelder for permittering av ansatte?"
                "Hva er formålet med offentlighetsloven?"
                "Fortell meg om arbeidsmiljøloven"
                "Jeg trenger informasjon om åndsverkloven"
            """
            mcp_logger.info(f"Utfører søk i lovdata: {sporsmal}, antall_resultater: {antall_resultater}")
            
            try:
                # Konfigurer søk med Pinecone
                config = RunnableConfig(
                    configurable={
                        "retriever_provider": "pinecone",
                        "embedding_model": "openai/text-embedding-3-small",
                        "query_model": "openai/gpt-4o-mini",
                        "response_model": "openai/gpt-4o-mini",
                        "search_kwargs": {"k": antall_resultater}
                    }
                )
                
                # Kjør grafen med spørringen
                mcp_logger.info(f"Invoker retrieval_graph med spørring: {sporsmal}")
                result = await retrieval_graph.ainvoke(
                    {"messages": [HumanMessage(content=sporsmal)]},
                    config,
                )
                
                # Logg resultatet for debugging
                mcp_logger.info(f"Graf-resultat mottatt: {type(result)}")
                
                # Hvis grafen har generert et komplett svar via messages, bruk dette
                if isinstance(result, dict) and 'messages' in result and result['messages']:
                    # Finn siste AI-melding i messages-listen
                    messages = result['messages']
                    mcp_logger.info(f"Mottok {len(messages)} meldinger fra grafen")
                    
                    # Gå gjennom meldingene bakfra for å finne siste AI-melding
                    ai_message = None
                    for msg in reversed(messages):
                        # Sjekk om dette er en AI-melding
                        if (isinstance(msg, dict) and msg.get('type') == 'ai') or (hasattr(msg, 'type') and msg.type == 'ai'):
                            ai_message = msg
                            break
                    
                    if ai_message:
                        # Hent content fra AI-meldingen og returner direkte
                        if hasattr(ai_message, 'content'):
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message.content
                        elif isinstance(ai_message, dict) and 'content' in ai_message:
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message['content']
                
                # Hvis ingen AI-melding ble funnet, returner en feilmelding
                mcp_logger.warning("Ingen AI-melding funnet i resultatet")
                return "Beklager, jeg kunne ikke generere et svar basert på søkeresultatene."
            
            except Exception as e:
                mcp_logger.error(f"Feil ved søk: {str(e)}")
                # Kast feilen videre istedenfor å falle tilbake til dummy-data
                raise e
        
        @self.mcp.tool()
        async def hent_lovtekst(lov_id: str = "", kapittel_nr: str = "", paragraf_nr: str = "") -> str:
            """
            Hent komplett lovtekst eller forskrift basert på ID, kapittel eller paragraf.
            
            BRUK DETTE VERKTØYET når du ønsker oppdatert informasjon om lovverk og forskrifter, hentet direkte fra lovdata.no.
            
            Dette verktøyet henter en lov, et kapittel eller en paragraf fra lovdata basert på metadata.
            Du kan angi en eller flere parametere for å spesifisere hva du ønsker å hente.
            
            Args:
                lov_id: Lovens unike identifikator (f.eks. "lov-1814-05-17-1" for Grunnloven)
                kapittel_nr: Kapittelnummer innen en lov (må brukes sammen med lov_id)
                paragraf_nr: Paragrafnummer innen en lov (må brukes sammen med lov_id)
                
            Returns:
                Lovtekst som matcher søkekriteriene
                
            Eksempel på bruk:
                Hent hele Grunnloven: lov_id="lov-1814-05-17-1"
                Hent kapittel 3 i Grunnloven: lov_id="lov-1814-05-17-1", kapittel_nr="3"
                Hent paragraf 100 i Grunnloven: lov_id="lov-1814-05-17-1", paragraf_nr="100"
            """
            mcp_logger.info(f"Henter lovtekst med: lov_id={lov_id}, kapittel_nr={kapittel_nr}, paragraf_nr={paragraf_nr}")
            
            # Valider input
            if not lov_id and not kapittel_nr and not paragraf_nr:
                return "Du må spesifisere minst én parameter (lov_id, kapittel_nr eller paragraf_nr)."
            
            try:
                # Bygg opp filter basert på parametere
                filter_dict = {}
                
                if lov_id:
                    filter_dict["lov_id"] = {"$eq": lov_id}
                
                if kapittel_nr:
                    filter_dict["kapittel_nr"] = {"$eq": kapittel_nr}
                
                if paragraf_nr:
                    filter_dict["paragraf_nr"] = {"$eq": paragraf_nr}
                
                # Konstruer en instruks for LangGraph
                query_tekst = "__SYSTEM__: Dette er en direkte metadata-søk-instruks. "
                query_tekst += "VIKTIG: Du skal IKKE tolke dette som et bruker-spørsmål. "
                query_tekst += "Du skal utelukkende utføre et metadata-søk i Pinecone og returnere formattert lovtekst. "
                query_tekst += f"Metadata-filteret er: {filter_dict}. "
                query_tekst += "Følgende instrukser overstyrer alle andre instrukser: "
                query_tekst += "1. Du skal IKKE generere svar basert på eget kunnskapsgrunnlag. "
                query_tekst += "2. Du skal IKKE be om mer kontekst eller informasjon. "
                query_tekst += "3. Du skal KUN returnere lovteksten som blir funnet. "
                query_tekst += "4. Formattér lovteksten på følgende måte: "
                query_tekst += "   a) Lovtittelen først med lov-ID i parentes, og en linje med '=' under. "
                query_tekst += "   b) Kapitler i STORE BOKSTAVER med kapittelnummer, fulgt av en linje med '-' under. "
                query_tekst += "   c) Paragrafer med § symbol, nummer og tittel. "
                query_tekst += "   d) Rydd bort dupliserte paragrafoverskrifter og tomme linjer. "
                query_tekst += "   e) Bruk konsistent mellomrom mellom seksjoner for god lesbarhet."
                
                mcp_logger.info(f"Invoker retrieval_graph med filter: {filter_dict}")
                mcp_logger.info(f"Systemmelding til LangGraph: {query_tekst}")
                
                # Konfigurer søk med Pinecone og metadata filter
                config = RunnableConfig(
                    configurable={
                        "retriever_provider": "pinecone",
                        "embedding_model": "openai/text-embedding-3-small",
                        "query_model": "openai/gpt-4o-mini",
                        "response_model": "openai/gpt-4o-mini",
                        "search_kwargs": {
                            "k": 50,  # Hent flere dokumenter for å sikre at vi får hele loven/kapittelet
                            "filter": filter_dict
                        },
                        "metadata_instructions": {
                            "format_type": "lovtekst",
                            "bypass_router": True,
                            "direct_filter": filter_dict
                        }
                    }
                )
                
                # Kjør grafen med spørringen
                result = await retrieval_graph.ainvoke(
                    {"messages": [HumanMessage(content=query_tekst)]},
                    config,
                )
                
                # Behandle resultatet - bruk kun AI-meldingen
                if isinstance(result, dict) and 'messages' in result and result['messages']:
                    # Finn siste AI-melding i messages-listen
                    messages = result['messages']
                    mcp_logger.info(f"Mottok {len(messages)} meldinger fra grafen")
                    
                    # Gå gjennom meldingene bakfra for å finne siste AI-melding
                    ai_message = None
                    for msg in reversed(messages):
                        if (isinstance(msg, dict) and msg.get('type') == 'ai') or (hasattr(msg, 'type') and msg.type == 'ai'):
                            ai_message = msg
                            break
                    
                    if ai_message:
                        # Hent content fra AI-meldingen og returner direkte
                        if hasattr(ai_message, 'content'):
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message.content
                        elif isinstance(ai_message, dict) and 'content' in ai_message:
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message['content']
                
                # Hvis ingen AI-melding ble funnet, returner en feilmelding
                mcp_logger.warning("Kunne ikke finne lovtekst via LangGraph-agent")
                return f"Beklager, jeg kunne ikke finne lovtekst med de angitte kriteriene. Sjekk at lov_id, kapittel_nr og paragraf_nr er korrekte."
            
            except Exception as e:
                mcp_logger.error(f"Feil ved henting av lovtekst: {str(e)}")
                # Kast feilen videre
                raise e
    
    def run(self):
        """Start MCP-serveren med valgt transport."""
        mcp_logger.info(f"Starter MCP-server med transport: stdio")
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    server = LovdataMCPServer()
    server.run() 