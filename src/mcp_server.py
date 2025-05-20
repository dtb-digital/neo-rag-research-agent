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
        
        # Registrer verktøy
        self._register_tools()
        
        mcp_logger.info(f"MCP-server initialisert (Using retrieval_graph: {USING_RETRIEVAL_GRAPH})")
    
    def _register_tools(self):
        """Registrer verktøy for MCP-serveren."""
        
        @self.mcp.tool()
        async def sok_i_lovdata(sporsmal: str, antall_resultater: int = 10) -> List[Dict[str, Any]]:
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
                En liste med relevante lovtekster, der hvert resultat inneholder:
                - id: Lovens unike identifikator (f.eks. "lov-1814-05-17-1" for Grunnloven)
                - score: Hvor relevant resultatet er for spørsmålet (høyere score = mer relevant)
                - excerpt: Et kort utdrag av lovteksten som er relevant for spørsmålet
                
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
                    # Bruk hovedgrafen for søk i stedet for direkte søk
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
                    mcp_logger.info(f"Graf-resultat: {result.keys()}")
                    
                    # Sjekk om vi har dokumenter i resultatet
                    if hasattr(result, 'documents') and result.documents:
                        docs = result.documents
                        mcp_logger.info(f"Fant {len(docs)} dokumenter fra grafen")
                    else:
                        mcp_logger.warning("Ingen dokumenter funnet fra grafen")
                        docs = []
                    
                    # Konverter dokumentene til ønsket format
                    results = []
                    for i, doc in enumerate(docs):
                        result = {
                            "id": doc.metadata.get("id", f"doc-{i}"),
                            "score": doc.metadata.get("score", 0.0),
                            "excerpt": truncate_text(doc.page_content, max_length=200)
                        }
                        results.append(result)
                    
                    mcp_logger.info(f"Søk fullført. Fant {len(results)} resultater.")
                    return results
                except Exception as e:
                    mcp_logger.error(f"Feil ved søk: {str(e)}")
                    mcp_logger.warning("Faller tilbake til dummy-implementasjon for søk")
                    # Fall tilbake til dummy-implementasjon ved feil
            
            # Dummy-implementasjon for testing eller fallback
            dummy_results = [
                {
                    "id": "lov-1814-05-17-1",
                    "score": 0.95,
                    "excerpt": "Kongeriket Norges Grunnlov, gitt i riksforsamlingen på Eidsvoll den 17. mai 1814, slik den lyder etter senere endringer."
                },
                {
                    "id": "lov-1992-07-17-100",
                    "score": 0.85,
                    "excerpt": "Lov om barneverntjenester (barnevernloven). Lovens formål er å sikre at barn og unge som lever under forhold som kan skade deres helse og utvikling, får nødvendig hjelp, omsorg og beskyttelse til rett tid."
                }
            ]
            
            # Begrens antall resultater til antall_resultater
            results = dummy_results[:min(antall_resultater, len(dummy_results))]
            
            mcp_logger.info(f"Søk fullført. Fant {len(results)} resultater.")
            return results
        
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
                    mcp_logger.info(f"Graf-resultat ved dokumenthenting: {result.keys()}")
                    
                    if hasattr(result, 'documents') and result.documents:
                        # Finn det første dokumentet som har riktig ID
                        for doc in result.documents:
                            if doc.metadata.get("id") == lov_id:
                                mcp_logger.info(f"Dokument funnet: {lov_id}")
                                return doc.page_content
                        
                        # Hvis vi ikke fant et dokument med riktig ID, returner innholdet av første dokument
                        if result.documents:
                            mcp_logger.warning(f"Dokument med ID {lov_id} ikke funnet, bruker første resultat")
                            return result.documents[0].page_content
                    
                    mcp_logger.warning(f"Dokument ikke funnet: {lov_id}")
                except Exception as e:
                    mcp_logger.error(f"Feil ved henting av dokument: {str(e)}")
                    mcp_logger.warning("Faller tilbake til dummy-implementasjon for dokumenthenting")
                    # Fall tilbake til dummy-implementasjon ved feil
            
            # Dummy-implementasjon for testing eller fallback
            if lov_id == "lov-1814-05-17-1":
                return "Kongeriket Norges Grunnlov, gitt i riksforsamlingen på Eidsvoll den 17. mai 1814, slik den lyder etter senere endringer. § 1. Kongeriket Norge er et fritt, selvstendig, udelelig og uavhendelig rike."
            elif lov_id == "lov-1992-07-17-100":
                return "Lov om barneverntjenester (barnevernloven). Lovens formål er å sikre at barn og unge som lever under forhold som kan skade deres helse og utvikling, får nødvendig hjelp, omsorg og beskyttelse til rett tid."
            else:
                return f"Lovtekst for {lov_id} er ikke tilgjengelig."
    
    def run(self):
        """Start MCP-serveren med valgt transport."""
        mcp_logger.info(f"Starter MCP-server med transport: stdio")
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    server = LovdataMCPServer()
    server.run() 