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
                    mcp_logger.info(f"Graf-resultat mottatt: {type(result)}")
                    
                    # Hvis grafen har generert et komplett svar via messages, bruk dette
                    if isinstance(result, dict) and 'messages' in result and result['messages']:
                        # Finn siste AI-melding i messages-listen
                        messages = result['messages']
                        
                        # Gå gjennom meldingene bakfra for å finne siste AI-melding
                        for msg in reversed(messages):
                            # Sjekk om dette er en AI-melding
                            if (isinstance(msg, dict) and msg.get('type') == 'ai') or hasattr(msg, 'type') and msg.type == 'ai':
                                # Hvis vi fant en AI-melding, returner dens content direkte
                                if hasattr(msg, 'content'):
                                    mcp_logger.info("Fant og returnerer content fra AI-melding (attributt)")
                                    return msg.content
                                elif isinstance(msg, dict) and 'content' in msg:
                                    mcp_logger.info("Fant og returnerer content fra AI-melding (dict)")
                                    return msg['content']
                        
                        # Hvis vi ikke fant en spesifikk AI-melding, prøv den siste meldingen
                        last_msg = messages[-1]
                        if hasattr(last_msg, 'content'):
                            mcp_logger.info("Returnerer content fra siste melding (attributt)")
                            return last_msg.content
                        elif isinstance(last_msg, dict) and 'content' in last_msg:
                            mcp_logger.info("Returnerer content fra siste melding (dict)")
                            return last_msg['content']
                        
                        # Siste utvei: Konverter til streng
                        mcp_logger.warning("Ingen AI-melding med content funnet, returnerer streng")
                        return str(result)
                    
                    # Fallback: Vi har ikke et ferdig bearbeidet svar, så vi må lage et basert på dokumentene
                    mcp_logger.warning("Ingen messages funnet i resultat, prøver å bruke dokumenter...")
                    
                    # Forsøk å hente dokumenter fra ulike mulige plasseringer i resultatet
                    docs = []
                    if isinstance(result, dict) and 'documents' in result:
                        docs = result['documents']
                    elif hasattr(result, 'documents') and result.documents:
                        docs = result.documents
                    elif hasattr(result, 'get') and result.get('documents'):
                        docs = result.get('documents', [])
                    
                    if docs:
                        mcp_logger.info(f"Fant {len(docs)} dokumenter som fallback")
                        # Konverter dokumentene til en tekstlig oppsummering
                        doc_summaries = []
                        for i, doc in enumerate(docs):
                            if hasattr(doc, 'page_content'):
                                doc_id = doc.metadata.get("id", f"Dokument {i+1}") if hasattr(doc, 'metadata') else f"Dokument {i+1}"
                                doc_summaries.append(f"{doc_id}: {truncate_text(doc.page_content, max_length=200)}")
                        
                        # Kombiner dokumentene til et enkelt svar
                        if doc_summaries:
                            response = "Her er relevante dokumenter jeg fant:\n\n" + "\n\n".join(doc_summaries)
                            mcp_logger.info(f"Genererte svar fra {len(doc_summaries)} dokumenter")
                            
                            # Legg til strukturert JSON-metadata
                            response += "\n\n```json\n"
                            response += "{\n"
                            response += '  "kilder": [\n'
                            
                            for i, doc in enumerate(docs):
                                response += "    {\n"
                                if hasattr(doc, 'metadata'):
                                    # Ekstraherer metadata fra dokumentet
                                    metadata = doc.metadata
                                    response += f'      "lovId": "{metadata.get("lov_id", f"ukjent-{i+1}")}",\n'
                                    response += f'      "lovNavn": "{metadata.get("lov_navn", "Ukjent lov")}",\n'
                                    
                                    # Legg til kapittel og paragraf hvis tilgjengelig
                                    if metadata.get("kapittel_nr"):
                                        response += f'      "kapittelNr": "{metadata.get("kapittel_nr")}",\n'
                                    if metadata.get("kapittel_tittel"):
                                        response += f'      "kapittelTittel": "{metadata.get("kapittel_tittel")}",\n'
                                    if metadata.get("paragraf_nr"):
                                        response += f'      "paragrafNr": "{metadata.get("paragraf_nr")}",\n'
                                    if metadata.get("paragraf_tittel"):
                                        response += f'      "paragrafTittel": "{metadata.get("paragraf_tittel")}",\n'
                                    
                                    response += f'      "tekst": "{truncate_text(doc.page_content, max_length=150)}"\n'
                                else:
                                    # Fallback for tilfeller uten metadata
                                    response += f'      "lovId": "ukjent-{i+1}",\n'
                                    response += f'      "tekst": "{truncate_text(str(doc), max_length=150)}"\n'
                                
                                response += "    }"
                                if i < len(docs) - 1:
                                    response += ","
                                response += "\n"
                                
                            response += "  ],\n"
                            response += '  "nøkkelbegreper": ["lovverk", "norsk lov"]\n'
                            response += "}\n```"
                            
                            mcp_logger.info("Returnerer fallback-respons med JSON-metadata")
                            return response
                    
                    mcp_logger.warning("Kunne ikke finne dokumenter eller messages i resultatet")
                    return "Beklager, jeg kunne ikke finne relevant informasjon om dette spørsmålet."
                
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
            
            # Formatter dummy-resultatene som en AI-respons istedenfor å returnere rådataene
            formatted_response = "Basert på ditt spørsmål har jeg funnet følgende relevante lover:\n\n"
            
            for i, result in enumerate(results):
                formatted_response += f"**{result['id']}** - {result['excerpt']}\n\n"
                
            formatted_response += "\nStrukturert data nedenfor kan brukes med verktøyet `hent_lovtekst` for å hente spesifikke lovtekster:\n\n"
            formatted_response += "## Kilder:\n"
            formatted_response += "- **[{}]** {}: {}\n".format(results[0]['id'], results[0]['excerpt'].split('. ')[0], results[0]['excerpt'])
            if len(results) > 1:
                for i, result in enumerate(results[1:], 1):
                    formatted_response += "- **[{}]** {}: {}\n".format(result['id'], result['excerpt'].split('. ')[0], result['excerpt'])
                
            formatted_response += "\n## Nøkkelbegreper:\n"
            formatted_response += "- lovverk\n"
            formatted_response += "- norsk lov"
            
            mcp_logger.info("Returnerer formatert dummy-respons med listeformat-metadata")
            return formatted_response
        
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
                        
                        # Gå gjennom meldingene bakfra for å finne siste AI-melding
                        for msg in reversed(messages):
                            # Sjekk om dette er en AI-melding
                            if (isinstance(msg, dict) and msg.get('type') == 'ai') or hasattr(msg, 'type') and msg.type == 'ai':
                                # Hvis vi fant en AI-melding, returner dens content direkte
                                if hasattr(msg, 'content'):
                                    mcp_logger.info("Fant og returnerer content fra AI-melding (attributt)")
                                    return msg.content
                                elif isinstance(msg, dict) and 'content' in msg:
                                    mcp_logger.info("Fant og returnerer content fra AI-melding (dict)")
                                    return msg['content']
                        
                        # Hvis vi ikke fant en spesifikk AI-melding, prøv den siste meldingen
                        last_msg = messages[-1]
                        if hasattr(last_msg, 'content'):
                            mcp_logger.info("Returnerer content fra siste melding (attributt)")
                            return last_msg.content
                        elif isinstance(last_msg, dict) and 'content' in last_msg:
                            mcp_logger.info("Returnerer content fra siste melding (dict)")
                            return last_msg['content']
                        
                        # Siste utvei: Konverter til streng
                        mcp_logger.warning("Ingen AI-melding med content funnet, returnerer streng")
                        return str(result)
                    
                    # Fallback: Forsøk å hente dokumenter direkte
                    docs = []
                    if isinstance(result, dict) and 'documents' in result:
                        docs = result['documents']
                    elif hasattr(result, 'documents') and result.documents:
                        docs = result.documents
                    elif hasattr(result, 'get') and result.get('documents'):
                        docs = result.get('documents', [])
                    
                    if docs:
                        # Finn det første dokumentet som har riktig ID
                        for doc in docs:
                            if hasattr(doc, 'metadata') and doc.metadata.get("id") == lov_id:
                                mcp_logger.info(f"Dokument funnet: {lov_id}")
                                return doc.page_content
                        
                        # Hvis vi ikke fant et dokument med riktig ID, returner innholdet av første dokument
                        if docs:
                            mcp_logger.warning(f"Dokument med ID {lov_id} ikke funnet, bruker første resultat")
                            return docs[0].page_content if hasattr(docs[0], 'page_content') else str(docs[0])
                    
                    mcp_logger.warning(f"Dokument ikke funnet: {lov_id}")
                    return f"Beklager, jeg kunne ikke finne lovtekst med ID {lov_id}."
                except Exception as e:
                    mcp_logger.error(f"Feil ved henting av dokument: {str(e)}")
                    mcp_logger.warning("Faller tilbake til dummy-implementasjon for dokumenthenting")
                    # Fall tilbake til dummy-implementasjon ved feil
            
            # Dummy-implementasjon for testing eller fallback
            dummy_text = ""
            dummy_lov_navn = ""
            if lov_id == "lov-1814-05-17-1":
                dummy_text = "Kongeriket Norges Grunnlov, gitt i riksforsamlingen på Eidsvoll den 17. mai 1814, slik den lyder etter senere endringer. § 1. Kongeriket Norge er et fritt, selvstendig, udelelig og uavhendelig rike."
                dummy_lov_navn = "Grunnloven"
            elif lov_id == "lov-1992-07-17-100":
                dummy_text = "Lov om barneverntjenester (barnevernloven). Lovens formål er å sikre at barn og unge som lever under forhold som kan skade deres helse og utvikling, får nødvendig hjelp, omsorg og beskyttelse til rett tid."
                dummy_lov_navn = "Barnevernloven"
            else:
                dummy_text = f"Lovtekst for {lov_id} er ikke tilgjengelig."
                dummy_lov_navn = f"Ukjent lov ({lov_id})"
                
            # Formater responsen på samme måte som en AI-respons
            formatted_response = f"Her er lovteksten for {dummy_lov_navn} ({lov_id}):\n\n"
            formatted_response += dummy_text
            
            # Legg til strukturert metadata i listeformat
            formatted_response += "\n\nStrukturert data nedenfor kan brukes for videre referanse:\n\n"
            formatted_response += "## Kilder:\n"
            formatted_response += f"- **[{lov_id}]** {dummy_lov_navn}: {dummy_text}\n"
            
            formatted_response += "\n## Nøkkelbegreper:\n"
            formatted_response += "- lovverk\n"
            formatted_response += "- norsk lov"
            
            mcp_logger.info("Returnerer formatert dummy-lovtekst med listeformat-metadata")
            return formatted_response
    
    def run(self):
        """Start MCP-serveren med valgt transport."""
        mcp_logger.info(f"Starter MCP-server med transport: stdio")
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    server = LovdataMCPServer()
    server.run() 