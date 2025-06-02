"""Tools for Neo RAG Research Agent.

Dette modulet inneholder fire selvstendige tools for juridisk informasjonssøk:
1. sok_lovdata - Grunnleggende vektorsøk i Pinecone
2. generer_sokestrenger - Intelligent oppbreking av komplekse spørsmål  
3. hent_lovtekst - Direkte henting av spesifikke lovtekster
4. sammenstill_svar - Sammenstilling av juridisk svar

Hver tool er komplett selvstendige uten kryss-referanser.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from pinecone import Pinecone


@tool
async def sok_lovdata(query: str, k: int = 5) -> List[Document]:
    """Søk i Lovdata med vektorsøk. Komplett, selvstedig implementasjon.
    
    Args:
        query: Søkestreng for juridisk informasjon
        k: Antall resultater som skal returneres
        
    Returns:
        Liste med relevante dokumenter med metadata
    """
    # Pinecone-oppsett (inspirert av MCP-server)
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index("lovdata-embedding-index")
    
    # Embed query (samme pattern som MCP-server)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = await embeddings.aembed_query(query)
    
    # Search Pinecone (kopierer søkelogikk)
    search_results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True
    )
    
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
    
    return documents


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
async def hent_lovtekst(lov_id: str, paragraf_nr: Optional[str] = None, kapittel_nr: Optional[str] = None) -> List[Document]:
    """Hent spesifikke lovtekster med metadata-filtering.
    
    Args:
        lov_id: Lovens ID (påkrevd)
        paragraf_nr: Spesifikk paragraf (valgfri)
        kapittel_nr: Spesifikt kapittel (valgfri)
        
    Returns:
        Liste med lovtekst-dokumenter med full metadata
    """
    # Samme Pinecone-oppsett som sok_lovdata
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index("lovdata-embedding-index")
    
    # Bygger filter (kopierer fra MCP-server hent_lovtekst())
    filter_dict = {"lov_id": {"$eq": lov_id}}
    if paragraf_nr:
        filter_dict["paragraf_nr"] = {"$eq": paragraf_nr}
    if kapittel_nr:
        filter_dict["kapittel_nr"] = {"$eq": kapittel_nr}
    
    # Søk med metadata-filter
    search_results = index.query(
        vector=[0] * 1536,  # Dummy vector for metadata-only søk
        top_k=50,           # Høyere k for komplette lovtekster
        filter=filter_dict,
        include_metadata=True
    )
    
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
    
    return documents


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