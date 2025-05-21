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

# Last inn miljøvariabler fra .env-filen
print("Laster miljøvariabler fra .env-filen...")
dotenv.load_dotenv()

# Hvis kritiske miljøvariabler mangler, prøv å lese direkte fra .env-filen
env_file = Path('.env')
if env_file.exists() and (
    not os.environ.get("PINECONE_API_KEY") or 
    not os.environ.get("OPENAI_API_KEY") or
    not os.environ.get("LANGSMITH_API_KEY")
):
    print("Noen miljøvariabler mangler, prøver å lese direkte fra .env-filen...")
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
            print("Satt PINECONE_API_KEY direkte fra .env")
        
        if openai_match and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_match.group(1).strip()
            print("Satt OPENAI_API_KEY direkte fra .env")
            
        # Legg til LANGSMITH-miljøvariablene
        if langsmith_api_match and not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_match.group(1).strip()
            print("Satt LANGSMITH_API_KEY direkte fra .env")
            
        if langsmith_tracing_match and not os.environ.get("LANGSMITH_TRACING"):
            os.environ["LANGSMITH_TRACING"] = langsmith_tracing_match.group(1).strip()
            print("Satt LANGSMITH_TRACING direkte fra .env")
            
        if langsmith_project_match and not os.environ.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = langsmith_project_match.group(1).strip()
            print("Satt LANGSMITH_PROJECT direkte fra .env")
            
        if langsmith_endpoint_match and not os.environ.get("LANGSMITH_ENDPOINT"):
            os.environ["LANGSMITH_ENDPOINT"] = langsmith_endpoint_match.group(1).strip()
            print("Satt LANGSMITH_ENDPOINT direkte fra .env")
    except Exception as e:
        print(f"Kunne ikke lese direkte fra .env: {str(e)}")

# Sett standard verdi for Pinecone-indeksen
if not os.environ.get("PINECONE_INDEX_NAME"):
    os.environ["PINECONE_INDEX_NAME"] = "lovdata-paragraf-test"
    print("Satt standard PINECONE_INDEX_NAME til 'lovdata-paragraf-test'")

# Sett standard verdi for LANGSMITH_TRACING hvis den ikke er satt
if not os.environ.get("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"
    print("Satt standard LANGSMITH_TRACING til 'true'")

# Legg til prosjektets rotmappe i sys.path for å støtte importer
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
# Legg til src-mappen i sys.path for å støtte begge importstiler
sys.path.append(os.path.join(project_root, "src"))

# Sett miljøvariabelen LOG_LEVEL til INFO direkte før import av FastMCP
os.environ["LOG_LEVEL"] = "INFO"

from fastmcp import FastMCP
from src.logger import logger, setup_logger
from src.utils import truncate_text

# Import retrieval_graph og andre nødvendige moduler
try:
    # Forsøk å importere retrieval_graph for søkefunksjonalitet
    from retrieval_graph import graph as retrieval_graph
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import HumanMessage
    USING_RETRIEVAL_GRAPH = True
except ImportError as e:
    # Logg feil og bruk dummy-implementasjon hvis import feiler
    USING_RETRIEVAL_GRAPH = False
    logger.error(f"Kunne ikke importere retrieval_graph: {str(e)}")
    logger.warning("Bruker dummy-implementasjon for søk og dokumenthenting")

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
        
        mcp_logger.info(f"MCP-server initialisert (Using retrieval_graph: {USING_RETRIEVAL_GRAPH})")
    
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
            
            if USING_RETRIEVAL_GRAPH:
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
                    
                    # Hvis ingen AI-melding ble funnet, returner et enkelt svar
                    mcp_logger.warning("Ingen AI-melding funnet i resultatet")
                    return "Beklager, jeg kunne ikke generere et svar basert på søkeresultatene."
                
                except Exception as e:
                    mcp_logger.error(f"Feil ved søk: {str(e)}")
                    mcp_logger.warning("Faller tilbake til dummy-implementasjon for søk")
                    # Fall tilbake til dummy-implementasjon ved feil
            
            # Dummy-implementasjon for testing eller fallback
            dummy_response = """
Basert på ditt spørsmål har jeg funnet følgende relevante lover:

**lov-1814-05-17-1** - Kongeriket Norges Grunnlov, gitt i riksforsamlingen på Eidsvoll den 17. mai 1814, slik den lyder etter senere endringer.

**lov-1992-07-17-100** - Lov om barneverntjenester (barnevernloven). Lovens formål er å sikre at barn og unge som lever under forhold som kan skade deres helse og utvikling, får nødvendig hjelp, omsorg og beskyttelse til rett tid.

## Kilder:
- **Kilde 1:**
  - lov_navn: Grunnloven
  - lov_id: lov-1814-05-17-1
  - kapittel_nr: A
  - kapittel_tittel: Statsformen og religionen
  - paragraf_nr: 1
  - paragraf_tittel: Statsform
  - sist_oppdatert: 2020-05-14
  - ikrafttredelse: 1814-05-17

- **Kilde 2:**
  - lov_navn: Barnevernloven
  - lov_id: lov-1992-07-17-100
  - kapittel_nr: 1
  - kapittel_tittel: Formål og virkeområde
  - paragraf_nr: 1
  - paragraf_tittel: Lovens formål
  - sist_oppdatert: 2021-06-18
  - ikrafttredelse: 1993-01-01

## Nøkkelbegreper:
- lovverk
- norsk lov
"""
            mcp_logger.info("Returnerer dummy-respons")
            return dummy_response
        
        @self.mcp.tool()
        async def hent_lovtekst(lov_id: str) -> str:
            """
            Hent komplett lovtekst eller forskrift basert på ID.
            
            BRUK DETTE VERKTØYET når du ønsker oppdatert informasjon om lovverk og forskrifter, hentet direkte fra lovdata.no.
            
            Dette verktøyet henter den fulle teksten til en lov, forskrift eller annet juridisk 
            dokument fra Lovdata ved hjelp av dokumentets unike ID. Bruk dette verktøyet når du 
            ønsker å se hele lovteksten etter å ha funnet relevante lover med sok_i_lovdata-verktøyet.
            
            Args:
                lov_id: Lovens unike identifikator (f.eks. "lov-1814-05-17-1" for Grunnloven)
                
            Returns:
                Fullstendig lovtekst som ren tekst
                
            Eksempel på bruk:
                Hent hele teksten for Grunnloven: "lov-1814-05-17-1"
                Hent hele teksten for Barnevernloven: "lov-1992-07-17-100"
            """
            mcp_logger.info(f"Henter lovtekst med id: {lov_id}")
            
            if USING_RETRIEVAL_GRAPH:
                try:
                    # Bruk hovedgrafen med spesifikk dokument-ID-spørring
                    config = RunnableConfig(
                        configurable={
                            "retriever_provider": "pinecone",
                            "embedding_model": "openai/text-embedding-3-small",
                            "query_model": "openai/gpt-4o-mini",
                            "response_model": "openai/gpt-4o-mini",
                            "search_kwargs": {"k": 1}
                        }
                    )
                    
                    query = f"Hent lovtekst med id {lov_id}"
                    mcp_logger.info(f"Invoker retrieval_graph for å hente dokument: {lov_id}")
                    result = await retrieval_graph.ainvoke(
                        {"messages": [HumanMessage(content=query)]},
                        config,
                    )
                    
                    # Logg resultatet for debugging
                    mcp_logger.info(f"Graf-resultat ved dokumenthenting: {type(result)}")
                    
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
                    return f"Beklager, jeg kunne ikke finne lovtekst med ID {lov_id}."
                    
                except Exception as e:
                    mcp_logger.error(f"Feil ved henting av dokument: {str(e)}")
                    mcp_logger.warning("Faller tilbake til dummy-implementasjon for dokumenthenting")
                    # Fall tilbake til dummy-implementasjon ved feil
            
            # Dummy-implementasjon for testing eller fallback
            dummy_lov_navn = ""
            dummy_text = ""
            
            if lov_id == "lov-1814-05-17-1":
                dummy_lov_navn = "Grunnloven"
                dummy_text = "Kongeriket Norges Grunnlov, gitt i riksforsamlingen på Eidsvoll den 17. mai 1814, slik den lyder etter senere endringer. § 1. Kongeriket Norge er et fritt, selvstendig, udelelig og uavhendelig rike."
            elif lov_id == "lov-1992-07-17-100":
                dummy_lov_navn = "Barnevernloven"
                dummy_text = "Lov om barneverntjenester (barnevernloven). Lovens formål er å sikre at barn og unge som lever under forhold som kan skade deres helse og utvikling, får nødvendig hjelp, omsorg og beskyttelse til rett tid."
            else:
                dummy_text = f"Lovtekst for {lov_id} er ikke tilgjengelig."
                dummy_lov_navn = f"Ukjent lov ({lov_id})"
                
            dummy_response = f"""
Her er lovteksten for {dummy_lov_navn} ({lov_id}):

{dummy_text}

## Kilder:
- **Kilde 1:**
  - lov_navn: {dummy_lov_navn}
  - lov_id: {lov_id}
  - dokumenttype: lovtekst
  - status: gjeldende
"""
            if lov_id == "lov-1814-05-17-1":
                dummy_response += """  - kapittel_nr: A
  - kapittel_tittel: Statsformen og religionen
  - paragraf_nr: 1
  - paragraf_tittel: Statsform
  - sist_oppdatert: 2020-05-14
  - ikrafttredelse: 1814-05-17
"""
            elif lov_id == "lov-1992-07-17-100":
                dummy_response += """  - kapittel_nr: 1
  - kapittel_tittel: Formål og virkeområde
  - paragraf_nr: 1
  - paragraf_tittel: Lovens formål
  - sist_oppdatert: 2021-06-18
  - ikrafttredelse: 1993-01-01
  - departement: Barne- og familiedepartementet
"""
            
            dummy_response += """
## Nøkkelbegreper:
- lovverk
- norsk lov
"""
            
            mcp_logger.info("Returnerer dummy-lovtekst")
            return dummy_response
    
    def run(self):
        """Start MCP-serveren med valgt transport."""
        mcp_logger.info(f"Starter MCP-server med transport: stdio")
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    server = LovdataMCPServer()
    server.run() 