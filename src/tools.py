"""Tools for Neo RAG Research Agent.

Dette modulet inneholder fire selvstendige tools for juridisk informasjonssøk:
1. sok_lovdata - Grunnleggende vektorsøk i Pinecone (oppdatert for mange treff)
2. generer_sokestrenger - Intelligent oppbreking av komplekse spørsmål  
3. hent_lovtekst - Direkte henting av spesifikke lovtekster
4. sammenstill_svar - Sammenstilling av juridisk svar

Bruker native LangGraph state management med Command objekter for automatisk
state-oppdatering via reduce_docs reducer.
"""

import asyncio
import os
from typing import Annotated, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import OpenAIEmbeddings
from langgraph.types import Command
from openai import AsyncOpenAI
from pinecone import Pinecone

from src.config import PINECONE_INDEX_NAME


@tool
async def sok_lovdata(
    query: str, 
    k: int = 10,
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """Søk i Lovdata med vektorsøk. Returnerer mange dokumenter for bedre dekning.
    
    Bruker native LangGraph state management - dokumenter oppdateres automatisk
    i state.documents via reduce_docs reducer.
    
    Args:
        query: Søkestreng for juridisk informasjon
        k: Antall resultater som skal returneres (standard: 10 for mange treff)
        
    Returns:
        Command som oppdaterer state.documents og returnerer ToolMessage
    """
    # Embed query asynkront
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = await embeddings.aembed_query(query)
    
    # Pinecone søk asynkront ved bruk av asyncio.to_thread
    def _sync_pinecone_search():
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        return index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )
    
    search_results = await asyncio.to_thread(_sync_pinecone_search)
    
    # Format til Document objekter (samme metadata-struktur)
    documents = []
    for match in search_results.matches:
        doc = Document(
            page_content=match.metadata.get("content", ""),
            metadata={
                "lov_id": match.metadata.get("lov_id"),
                "paragraf_nr": match.metadata.get("paragraf_nr"), 
                "kapittel_nr": match.metadata.get("kapittel_nr"),
                "lov_navn": match.metadata.get("lov_tittel"),
                "score": match.score
            }
        )
        documents.append(doc)
    
    # Lag feedback melding
    result_summary = f"Søk fullført for '{query}'. Fant {len(documents)} relevante dokumenter fra Lovdata."
    
    if documents:
        # Legg til sammendrag av hva som ble funnet
        unique_laws = set()
        for doc in documents[:5]:  # Vis de 5 første
            if doc.metadata.get("lov_navn"):
                unique_laws.add(doc.metadata["lov_navn"])
        
        if unique_laws:
            laws_text = ", ".join(list(unique_laws)[:3])
            result_summary += f" Inkluderer dokumenter fra: {laws_text}"
            if len(unique_laws) > 3:
                result_summary += f" og {len(unique_laws) - 3} andre lover."
    
    result_summary += " Dokumentene er lagt til i agent state for videre analyse."
    
    # Returner Command som oppdaterer state.documents automatisk
    return Command(
        update={
            "documents": documents,
            "messages": [ToolMessage(content=result_summary, tool_call_id=tool_call_id)]
        }
    )


@tool
async def generer_sokestrenger(question: str, num_queries: int = 3) -> List[str]:
    """Generer flere søkestrenger fra ett spørsmål. Komplett implementasjon.
    
    Args:
        question: Brukerens opprinnelige spørsmål
        num_queries: Ønsket antall søkestrenger
        
    Returns:
        Liste med genererte, varierte søkestrenger
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": """Du genererer varierte søkestrenger for juridisk informasjon i norsk lovdata.
                
Opprett søkestrenger som:
- Dekker forskjellige aspekter av spørsmålet
- Bruker varierende juridiske termer
- Er spesifikke nok til å finne relevante lover
- Unngår for brede søkeord

Skriv hver søkestreng på egen linje, kun søkestrengene uten nummerering eller punkter."""
            },
            {
                "role": "user", 
                "content": f"Lag {num_queries} forskjellige søkestrenger for: {question}"
            }
        ]
    )
    
    # Parse respons til liste med strenger
    content = response.choices[0].message.content
    queries = [q.strip('- ').strip() for q in content.split('\n') if q.strip()]
    return queries[:num_queries]  # Sørg for riktig antall


@tool
async def hent_lovtekst(
    lov_id: str, 
    paragraf_nr: Optional[str] = None, 
    kapittel_nr: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """Hent spesifikke lovtekster med metadata-filtering.
    
    Bruker native LangGraph state management - dokumenter oppdateres automatisk
    i state.documents via reduce_docs reducer.
    
    Args:
        lov_id: Lovens ID (påkrevd)
        paragraf_nr: Spesifikk paragraf (valgfri)
        kapittel_nr: Spesifikt kapittel (valgfri)
        
    Returns:
        Command som oppdaterer state.documents og returnerer ToolMessage
    """
    # Bygger filter
    filter_dict = {"lov_id": {"$eq": lov_id}}
    if paragraf_nr:
        filter_dict["paragraf_nr"] = {"$eq": paragraf_nr}
    if kapittel_nr:
        filter_dict["kapittel_nr"] = {"$eq": kapittel_nr}
    
    # Pinecone søk asynkront
    def _sync_pinecone_filter_search():
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        return index.query(
            vector=[0] * 1536,  # Dummy vector for metadata-only søk
            top_k=50,           # Høyere k for komplette lovtekster
            filter=filter_dict,
            include_metadata=True
        )
    
    search_results = await asyncio.to_thread(_sync_pinecone_filter_search)
    
    # Samme dokumentformatering som sok_lovdata
    documents = []
    for match in search_results.matches:
        doc = Document(
            page_content=match.metadata.get("content", ""),
            metadata={
                "lov_id": match.metadata.get("lov_id"),
                "paragraf_nr": match.metadata.get("paragraf_nr"),
                "kapittel_nr": match.metadata.get("kapittel_nr"),
                "lov_navn": match.metadata.get("lov_tittel")
            }
        )
        documents.append(doc)
    
    # Lag feedback melding
    filter_desc = f"lov_id={lov_id}"
    if paragraf_nr:
        filter_desc += f", paragraf_nr={paragraf_nr}"
    if kapittel_nr:
        filter_desc += f", kapittel_nr={kapittel_nr}"
    
    result_summary = f"Hentet {len(documents)} dokumenter for {filter_desc}. Dokumentene er lagt til i agent state for videre analyse."
    
    # Returner Command som oppdaterer state.documents automatisk
    return Command(
        update={
            "documents": documents,
            "messages": [ToolMessage(content=result_summary, tool_call_id=tool_call_id)]
        }
    )


@tool
async def sammenstill_svar(documents: List[Document], original_question: str) -> str:
    """Sammenstill juridisk svar basert på dokumenter. Komplett implementasjon.
    
    Args:
        documents: Alle relevante dokumenter fra søk
        original_question: Brukerens opprinnelige spørsmål
        
    Returns:
        Ferdig formulert juridisk svar med kildehenvisninger
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Format dokumenter for prompt (inspirert av MCP respond())
    docs_text = ""
    for i, doc in enumerate(documents):
        metadata_str = ", ".join([
            f"{k}: {v}" for k, v in doc.metadata.items() 
            if k in ["lov_id", "lov_navn", "paragraf_nr", "kapittel_nr"] and v
        ])
        docs_text += f"\n\nDokument {i+1}:\n{doc.page_content}\nMetadata: {metadata_str}"
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Du er en juridisk assistent som gir presise svar basert på norsk lovgivning.

Oppgaver:
- Gi strukturerte, juridisk korrekte svar
- Inkluder relevante kildehenvisninger 
- Referer til spesifikke paragrafer når relevant
- Hvis informasjonen er utilstrekkelig, kommuniser dette tydelig
- Skriv på norsk med klar, juridisk terminologi

Format svaret med:
1. Direkte svar på spørsmålet
2. Juridisk begrunnelse
3. Relevante lovparagrafer og kilder
4. Eventuelle forbehold eller presiseringer"""
            },
            {
                "role": "user",
                "content": f"Spørsmål: {original_question}\n\nRelevante juridiske dokumenter:{docs_text}\n\nGi et strukturert juridisk svar med kildehenvisninger."
            }
        ]
    )
    
    return response.choices[0].message.content


# Liste med alle tools for enkel import
TOOLS = [sok_lovdata, generer_sokestrenger, hent_lovtekst, sammenstill_svar] 