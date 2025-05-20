#!/usr/bin/env python
"""
Ingest script for Lovdata RAG-agent.

Dette scriptet laster inn lovtekster fra en URL til Pinecone vektorbasen.
Det bruker LLM for å ekstraherer metadata fra lovteksten.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Tredjeparts biblioteker
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Import Pinecone - støtter både ny og gammel importstil
try:
    # Ny importstil (pinecone pakke)
    from pinecone import Pinecone, ServerlessSpec
    USING_NEW_PINECONE = True
except ImportError:
    # Gammel importstil (pinecone-client pakke)
    import pinecone
    USING_NEW_PINECONE = False

# For kompatibilitet med langchain_pinecone
# Midlertidig løsning mens langchain oppdaterer sine avhengigheter
import sys
import types
if USING_NEW_PINECONE and "pinecone" not in sys.modules:
    sys.modules["pinecone"] = types.ModuleType("pinecone")
    sys.modules["pinecone"].Pinecone = Pinecone
    sys.modules["pinecone"].ServerlessSpec = ServerlessSpec

# Nå kan vi importere langchain_pinecone
from langchain_pinecone import PineconeVectorStore

# Legg til prosjektets rotmappe i sys.path for å støtte importer
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Oppsett av logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ingest-lovdata")

# Sjekk at nødvendige miljøvariabler er satt
required_env_vars = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT"
]

for var in required_env_vars:
    if not os.environ.get(var):
        logger.error(f"Miljøvariabel {var} er ikke satt")
        sys.exit(1)


def fetch_document(url: str) -> str:
    """
    Hent HTML-innhold fra en URL.
    
    Args:
        url: URL til lovdokumentet
        
    Returns:
        HTML-innhold som string
    """
    logger.info(f"Henter dokument fra {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Feil ved henting av dokument: {str(e)}")
        raise


def clean_html(html: str) -> str:
    """
    Rens HTML og ekstraher hovedinnholdet.
    
    Args:
        html: HTML-innhold
        
    Returns:
        Renset tekst
    """
    logger.info("Renser HTML og ekstraherer tekst")
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Fjern script og style elementer
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Forsøk å finne hovedinnholdet (lovdata-spesifikk)
        main_content = None
        
        # Vanlige selektorer for hovedinnhold på lovdata.no
        content_selectors = [
            "div.lovdata-document-content",
            "div#main-content",
            "article.dokument",
            "div.dokument",
            "main"
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                main_content = content
                break
        
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            # Fallback til hele body
            text = soup.body.get_text(separator="\n", strip=True)
        
        # Fjern ekstra whitespace
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)
        
        return text
    except Exception as e:
        logger.error(f"Feil ved rensing av HTML: {str(e)}")
        raise


def extract_metadata_with_llm(text: str) -> Dict[str, Any]:
    """
    Bruk LLM for å ekstraherer metadata fra lovteksten.
    
    Args:
        text: Renset lovtekst
        
    Returns:
        Strukturert metadata som dict
    """
    logger.info("Ekstraherer metadata med LLM")
    
    # Begrens teksten for LLM (ta de første 8000 tegnene)
    # Dette skulle være nok for å identifisere metadata
    limited_text = text[:8000]
    
    # Definer prompt for LLM
    prompt_template = """
    Du er en spesialist på norsk lovdata. Analyser følgende lovtekst og ekstraher strukturert metadata.
    Vær så presis som mulig, og returner informasjonen i det eksakte JSON-formatet som er forespurt.
    
    Lovtekst:
    ```
    {text}
    ```
    
    Ekstraher følgende informasjon i JSON-format:
    1. lov_id: Lovens nummer og dato (f.eks. "LOV-1814-05-17-1")
    2. tittel: Lovens fulle tittel
    3. korttittel: Lovens korttittel (hvis tilgjengelig)
    4. ikrafttredelse: Dato loven trådte i kraft
    5. sist_endret: Dato for siste endring (hvis tilgjengelig)
    6. struktur: Overordnet struktur (kapitler, deler, osv.)
    
    Returner kun et gyldig JSON-objekt med disse nøklene, ingen annen tekst.
    Eksempel:
    {{
        "lov_id": "LOV-1814-05-17-1",
        "tittel": "Kongeriket Norges Grunnlov",
        "korttittel": "Grunnloven",
        "ikrafttredelse": "1814-05-17",
        "sist_endret": "2020-05-01",
        "struktur": ["Kapittel A", "Kapittel B", "Kapittel C"]
    }}
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    # Opprett LLM og chain
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # Kjør LLM for å ekstrahere metadata
        result = chain.run(text=limited_text)
        
        # Parse JSON resultat
        metadata = json.loads(result.strip())
        
        # Legg til opprinnelig URL som metadata
        metadata["source_text"] = text
        
        logger.info(f"Ekstrahert metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
        return metadata
    except Exception as e:
        logger.error(f"Feil ved ekstrahering av metadata: {str(e)}")
        # Returner et minimalt metadata-sett hvis LLM feiler
        return {
            "lov_id": "ukjent",
            "tittel": "Ukjent lovdokument",
            "korttittel": "",
            "ikrafttredelse": "",
            "sist_endret": "",
            "struktur": [],
            "source_text": text
        }


def split_into_chunks(metadata: Dict[str, Any]) -> List[Document]:
    """
    Del opp lovteksten i mindre chunks for vektorisering.
    
    Args:
        metadata: Strukturert metadata med source_text
        
    Returns:
        Liste av Document-objekter med metadata
    """
    logger.info("Deler teksten i chunks")
    
    text = metadata.pop("source_text")  # Fjern kildetekst fra metadata
    
    # Opprett text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Del opp teksten
    chunks = text_splitter.split_text(text)
    logger.info(f"Teksten er delt i {len(chunks)} chunks")
    
    # Opprett Document-objekter med metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Kopier metadata for hvert chunk
        chunk_metadata = metadata.copy()
        
        # Legg til chunk-spesifikk metadata
        chunk_metadata["chunk_id"] = i
        chunk_metadata["id"] = f"{metadata.get('lov_id', 'ukjent')}-chunk-{i}"
        
        # Opprett Document
        doc = Document(
            page_content=chunk,
            metadata=chunk_metadata
        )
        documents.append(doc)
    
    return documents


def load_into_pinecone(documents: List[Document], index_name: str = "lovdata-index", batch_size: int = 100) -> None:
    """
    Last opp dokumenter til Pinecone.
    
    Args:
        documents: Liste av Document-objekter
        index_name: Navn på Pinecone-indeksen
        batch_size: Antall dokumenter per batch
    """
    logger.info(f"Laster {len(documents)} dokumenter til Pinecone-indeks '{index_name}'")
    
    # Initialiser Pinecone
    if USING_NEW_PINECONE:
        # Ny API (pinecone pakke)
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        
        # Sjekk om indeksen eksisterer
        try:
            # Forsøk å få indeksen
            index = pc.Index(index_name)
            logger.info(f"Kobler til eksisterende Pinecone-indeks: {index_name}")
        except Exception:
            # Indeksen eksisterer ikke, opprett ny
            logger.info(f"Oppretter ny Pinecone-indeks: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimensjon for OpenAI embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=os.environ["PINECONE_ENVIRONMENT"].split("-aws")[0])
            )
            index = pc.Index(index_name)
    else:
        # Gammel API (pinecone-client pakke)
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"]
        )
        
        # Sjekk om indeksen eksisterer
        if index_name not in pinecone.list_indexes():
            logger.info(f"Oppretter ny Pinecone-indeks: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # Dimensjon for OpenAI embeddings
                metric="cosine"
            )
    
    # Opprett embeddings-modell
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Last dokumenter i batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logger.info(f"Laster batch {i//batch_size + 1}/{len(documents)//batch_size + 1} ({len(batch)} dokumenter)")
        
        try:
            # Opprett eller koble til eksisterende vektorstore
            vector_store = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=index_name
            )
            logger.info(f"Batch {i//batch_size + 1} lastet inn vellykket")
        except Exception as e:
            logger.error(f"Feil ved innlasting av batch {i//batch_size + 1}: {str(e)}")
            # Fortsett med neste batch


def main():
    """Hovedfunksjon"""
    parser = argparse.ArgumentParser(description="Last inn lovtekster fra URL til Pinecone vektorbasen")
    parser.add_argument("--url", required=True, help="URL til lovdokumentet")
    parser.add_argument("--index-name", default="lovdata-index", help="Navn på Pinecone-indeksen (default: lovdata-index)")
    parser.add_argument("--batch-size", type=int, default=100, help="Antall dokumenter per batch (default: 100)")
    
    args = parser.parse_args()
    
    try:
        # Sett Pinecone-indeksnavn i miljøvariabel for bruk av andre moduler
        os.environ["PINECONE_INDEX_NAME"] = args.index_name
        
        # Kjør ingestion pipeline
        start_time = time.time()
        
        html = fetch_document(args.url)
        text = clean_html(html)
        structured_data = extract_metadata_with_llm(text)
        documents = split_into_chunks(structured_data)
        load_into_pinecone(documents, args.index_name, args.batch_size)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Ingestion fullført! Tid brukt: {elapsed_time:.2f} sekunder")
        logger.info(f"Lastet inn {len(documents)} dokumenter i Pinecone-indeks '{args.index_name}'")
        
    except Exception as e:
        logger.exception(f"Feil i ingestion-prosessen: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 