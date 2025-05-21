#!/usr/bin/env python
"""
Ingest script for Lovdata RAG-agent.

Dette scriptet laster inn lovtekster fra en URL til Pinecone vektorbasen.
Det bruker mønstergjenkjenning for å ekstraherer metadata og struktur fra lovteksten.
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

# Tredjeparts biblioteker
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

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
    "PINECONE_API_KEY"
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


def parse_lov_metadata(text: str) -> Dict[str, Any]:
    """
    Ekstraher metadata fra lovtekst ved hjelp av regulære uttrykk.
    
    Args:
        text: Renset lovtekst
        
    Returns:
        Strukturert metadata som dict
    """
    logger.info("Ekstraherer metadata med mønstergjenkjenning")
    
    # Initialiser metadata dictionary
    metadata = {
        "lov_id": "ukjent",
        "lov_navn": "ukjent",
        "lov_tittel": "Ukjent lovdokument",
        "ikrafttredelse": "",
        "sist_endret": "",
        "språk": "nob",  # Default til bokmål
        "status": "gjeldende",
        "source_text": text
    }
    
    # Finn lovens identifikasjon
    lov_id_match = re.search(r'LOV-(\d{4}-\d{2}-\d{2}-\d+)', text)
    if lov_id_match:
        metadata['lov_id'] = f"lov-{lov_id_match.group(1).lower()}"
    
    # Finn lovens korttittel
    korttittel_match = re.search(r'Korttittel[:\s]+([^\n–]+)', text)
    if korttittel_match:
        metadata['lov_navn'] = korttittel_match.group(1).strip().replace('–', '').split('–')[0].strip()
    
    # Finn lovens fulle tittel
    # Prøv flere varianter av tittel-mønstre
    tittel_patterns = [
        r'Lov om ([^\n]+?)\s*\n',
        r'Lov om rett til ([^\n]+?)\s*\n',
        r'Lov [^\n]+? \((.*?)\)',
    ]
    
    for pattern in tittel_patterns:
        tittel_match = re.search(pattern, text, re.IGNORECASE)
        if tittel_match:
            metadata['lov_tittel'] = f"Lov om {tittel_match.group(1).strip()}"
            break
    
    # Prøv å finne tittelen basert på overskrift (fallback)
    if metadata['lov_tittel'] == "Ukjent lovdokument":
        lines = text.split('\n')
        for line in lines[:10]:  # Sjekk de første 10 linjene
            if "lov om" in line.lower() and len(line) < 150:  # Unngå for lange linjer
                metadata['lov_tittel'] = line.strip()
                break
    
    # Finn ikrafttredelse
    ikraft_patterns = [
        r'Ikrafttredelse\s+(\d{2}\.\d{2}\.\d{4})',
        r'Ikrafttredelse\s+(\d{1,2}\s+\w+\s+\d{4})',
    ]
    
    for pattern in ikraft_patterns:
        ikraft_match = re.search(pattern, text)
        if ikraft_match:
            dato_str = ikraft_match.group(1).strip()
            # Konverter til ISO-format hvis mulig
            try:
                if '.' in dato_str:
                    # Format: DD.MM.YYYY
                    dato_deler = dato_str.split('.')
                    if len(dato_deler) == 3:
                        metadata['ikrafttredelse'] = f"{dato_deler[2]}-{dato_deler[1]}-{dato_deler[0]}"
                # Andre datoformater kan implementeres ved behov
            except:
                # Behold originalt format hvis konvertering feiler
                metadata['ikrafttredelse'] = dato_str
            break
    
    # Finn sist endret
    endret_patterns = [
        r'Sist endret\s+(\d{2}\.\d{2}\.\d{4})',
        r'Sist endret\s+(\d{1,2}\s+\w+\s+\d{4})',
    ]
    
    for pattern in endret_patterns:
        endret_match = re.search(pattern, text)
        if endret_match:
            dato_str = endret_match.group(1).strip()
            try:
                if '.' in dato_str:
                    dato_deler = dato_str.split('.')
                    if len(dato_deler) == 3:
                        metadata['sist_endret'] = f"{dato_deler[2]}-{dato_deler[1]}-{dato_deler[0]}"
                # Andre datoformater kan implementeres ved behov
            except:
                metadata['sist_endret'] = dato_str
            break
    
    # Bestem språk basert på tekst-innhold
    if "bokmål" in text.lower() or "lov om" in text.lower():
        metadata["språk"] = "nob"
    elif "nynorsk" in text.lower() or "lov om rett til" in text.lower():
        metadata["språk"] = "nno"
    
    # Generer en UUID for lovreferansen
    metadata["uuid"] = str(uuid.uuid4())
    
    logger.info(f"Ekstrahert metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
    return metadata


def chunke_basert_pa_paragrafer(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """
    Del opp lovteksten basert på paragrafer og kapitler.
    
    Args:
        text: Rå lovtekst
        metadata: Grunnleggende metadata for loven
        
    Returns:
        Liste av Document-objekter med berikede metadata
    """
    logger.info("Deler teksten basert på paragrafer og struktur")
    
    # Regex for å identifisere kapitler og paragrafer
    kapittel_monster = r'(?:\n|\A)\s*Kapittel\s+(\d+)[A-Z]?\.?\s*([^\n]+)'
    paragraf_monster = r'(?:\n|\A)\s*§\s*(\d+)\.?\s*([^\n]*)'
    
    # Finn alle kapitler
    kapitler = list(re.finditer(kapittel_monster, text))
    
    # Finn alle paragrafer
    paragrafer = list(re.finditer(paragraf_monster, text))
    
    logger.info(f"Fant {len(kapitler)} kapitler og {len(paragrafer)} paragrafer")
    
    if len(paragrafer) == 0:
        logger.warning("Ingen paragrafer funnet, faller tilbake til standard chunking")
        return standard_chunking(text, metadata)
    
    # Opprett dokumenter for hver paragraf
    documents = []
    
    # Funksjon for å finne hvilket kapittel en paragraf tilhører
    def finn_kapittel(paragraf_pos):
        for i, kap in enumerate(kapitler):
            if kap.start() < paragraf_pos:
                if i + 1 < len(kapitler) and kapitler[i + 1].start() > paragraf_pos:
                    return kap
                elif i + 1 == len(kapitler):
                    return kap
        return None
    
    # Behandle hver paragraf
    for i, paragraf in enumerate(paragrafer):
        paragraf_nr = paragraf.group(1)
        paragraf_tittel = paragraf.group(2).strip()
        
        # Bestem start og slutt for paragraf-tekst
        start_pos = paragraf.end()
        end_pos = len(text)
        
        # Hvis det er flere paragrafer, sett slutt til starten av neste paragraf
        if i + 1 < len(paragrafer):
            end_pos = paragrafer[i + 1].start()
        
        # Hent paragraf-tekst
        paragraf_tekst = text[start_pos:end_pos].strip()
        
        # Finn kapittel for denne paragrafen
        kapittel = finn_kapittel(paragraf.start())
        kapittel_nr = ""
        kapittel_tittel = ""
        
        if kapittel:
            kapittel_nr = kapittel.group(1)
            kapittel_tittel = kapittel.group(2).strip()
        
        # Lag chunk-metadata
        chunk_metadata = metadata.copy()
        del chunk_metadata['source_text']  # Fjern kildetekst fra metadata
        
        # Legg til struktur-metadata
        chunk_metadata.update({
            "kapittel_nr": kapittel_nr,
            "kapittel_tittel": kapittel_tittel,
            "paragraf_nr": paragraf_nr,
            "paragraf_tittel": paragraf_tittel,
            "chunk_type": "paragraf",
            "chunk_index": i,
            "chunk_id": i,
            "id": f"{metadata.get('lov_id', 'ukjent')}-paragraf-{paragraf_nr}",
            "parent_id": f"kapittel-{kapittel_nr}" if kapittel_nr else None
        })
        
        # Lag komplett tekst med kontekst
        full_tekst = f"§ {paragraf_nr}. {paragraf_tittel}\n\n{paragraf_tekst}"
        
        # Opprett Document
        doc = Document(
            page_content=full_tekst,
            metadata=chunk_metadata
        )
        documents.append(doc)
    
    logger.info(f"Opprettet {len(documents)} paragraph-baserte chunks")
    return documents


def standard_chunking(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """
    Fallback til standard chunking hvis paragraf-basert chunking ikke fungerer.
    
    Args:
        text: Rå lovtekst
        metadata: Grunnleggende metadata for loven
        
    Returns:
        Liste av Document-objekter
    """
    logger.info("Bruker standard chunking (fallback)")
    
    # Del teksten basert på logiske skiller
    chunks = []
    lines = text.split("\n")
    
    current_chunk = []
    current_size = 0
    target_size = 1000
    
    for line in lines:
        line_length = len(line)
        
        if current_size + line_length > target_size and current_chunk:
            # Denne linjen vil gjøre chunken for stor, lagre chunken og start en ny
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_length
        else:
            # Legg til linjen i nåværende chunk
            current_chunk.append(line)
            current_size += line_length
    
    # Legg til siste chunk hvis det er noe igjen
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    logger.info(f"Teksten er delt i {len(chunks)} standard chunks")
    
    # Opprett Document-objekter med metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Kopier metadata for hvert chunk
        chunk_metadata = metadata.copy()
        del chunk_metadata['source_text']  # Fjern kildetekst fra metadata
        
        # Legg til chunk-spesifikk metadata
        chunk_metadata.update({
            "chunk_id": i,
            "id": f"{metadata.get('lov_id', 'ukjent')}-chunk-{i}",
            "chunk_type": "tekst",
            "chunk_index": i
        })
        
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
    
    # Initialiser Pinecone med ny syntax
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Sjekk om indeksen eksisterer
    if index_name not in pc.list_indexes().names():
        logger.info(f"Oppretter ny Pinecone-indeks: {index_name}")
        pc.create_index(
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
    parser.add_argument("--force-standard-chunking", action="store_true", 
                        help="Tving bruk av standard chunking i stedet for paragraf-basert")
    
    args = parser.parse_args()
    
    try:
        # Sett Pinecone-indeksnavn i miljøvariabel for bruk av andre moduler
        os.environ["PINECONE_INDEX_NAME"] = args.index_name
        
        # Kjør ingestion pipeline
        start_time = time.time()
        
        html = fetch_document(args.url)
        text = clean_html(html)
        metadata = parse_lov_metadata(text)
        
        # Velg chunking-metode
        if args.force_standard_chunking:
            documents = standard_chunking(text, metadata)
        else:
            documents = chunke_basert_pa_paragrafer(text, metadata)
        
        load_into_pinecone(documents, args.index_name, args.batch_size)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Ingestion fullført! Tid brukt: {elapsed_time:.2f} sekunder")
        logger.info(f"Lastet inn {len(documents)} dokumenter i Pinecone-indeks '{args.index_name}'")
        
    except Exception as e:
        logger.exception(f"Feil i ingestion-prosessen: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 