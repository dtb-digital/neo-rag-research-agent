#!/usr/bin/env python
"""
Test Pinecone-tilkoblingen direkte.

Dette scriptet tester om vi kan koble til Pinecone-indeksen og utføre et enkelt vektorsøk.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path

# Last inn miljøvariabler fra .env-filen
load_dotenv()

# Legg til prosjektets rotmappe i sys.path for å støtte importer
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Oppsett av logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test-pinecone")

# Sjekk at nødvendige miljøvariabler er satt
required_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        logger.error(f"{var} er ikke satt. Dette vil føre til autentiseringsfeil.")
        sys.exit(1)

# Sett standard Pinecone-konfigurasjon hvis ikke allerede satt
if "PINECONE_INDEX_NAME" not in os.environ:
    os.environ["PINECONE_INDEX_NAME"] = "lovdata-index"
    logger.info(f"Satt PINECONE_INDEX_NAME til: {os.environ['PINECONE_INDEX_NAME']}")

if "PINECONE_ENVIRONMENT" not in os.environ:
    os.environ["PINECONE_ENVIRONMENT"] = "us-east-1-aws"
    logger.info(f"Satt PINECONE_ENVIRONMENT til: {os.environ['PINECONE_ENVIRONMENT']}")

# Spesifikk host for Pinecone-indeksen
pinecone_host = "lovdata-index-111be8b.svc.aped-4627-b74a.pinecone.io"
if "PINECONE_HOST" not in os.environ and pinecone_host:
    os.environ["PINECONE_HOST"] = pinecone_host
    logger.info(f"Satt PINECONE_HOST til: {os.environ['PINECONE_HOST']}")


def test_pinecone_connection():
    """Test om vi kan koble til Pinecone-indeksen."""
    import pinecone
    
    logger.info("Tester tilkobling til Pinecone...")
    
    try:
        # Initialiser med pinecone-client 5.0.1 API
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        logger.info("Pinecone-klient opprettet")
        
        index_name = os.environ["PINECONE_INDEX_NAME"]
        host = os.environ.get("PINECONE_HOST")
        
        # Prøv å koble til indeksen
        if host:
            logger.info(f"Kobler til indeks via direkte host: {host}")
            index = pc.Index(host=host)
        else:
            logger.info(f"Kobler til indeks via indeksnavn: {index_name}")
            index_info = pc.describe_index(name=index_name)
            index = pc.Index(host=index_info.host)
        
        # Vis statistikk
        stats = index.describe_index_stats()
        logger.info(f"Pinecone-indeksstatistikk: {stats}")
        logger.info("Vellykket tilkobling til Pinecone")
        
        return True
    except Exception as e:
        logger.error(f"Feil ved tilkobling til Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pinecone_search(query: str, top_k: int = 5):
    """Test vektorsøk i Pinecone."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    import pinecone
    
    logger.info(f"Tester søk i Pinecone med spørringen: '{query}'")
    
    try:
        # Opprett embedding-modell
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("OpenAI embedding-modell opprettet")
        
        # Koble til Pinecone
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        
        # Velg indeksen
        index_name = os.environ["PINECONE_INDEX_NAME"]
        host = os.environ.get("PINECONE_HOST")
        
        if host:
            logger.info(f"Kobler til indeks via direkte host: {host}")
            index = pc.Index(host=host)
        else:
            logger.info(f"Kobler til indeks via indeksnavn: {index_name}")
            index_info = pc.describe_index(name=index_name)
            index = pc.Index(host=index_info.host)
            
        # Opprett vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="page_content"
        )
        logger.info("PineconeVectorStore opprettet")
        
        # Opprett retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        logger.info(f"Retriever opprettet med k={top_k}")
        
        # Utfør søk
        query_embedding = embedding_model.embed_query(query)
        logger.info(f"Query embedding opprettet, dimensjon: {len(query_embedding)}")
        
        # Utfør søk med retriever
        docs = retriever.invoke(query)
        logger.info(f"Søk fullført, fant {len(docs)} dokumenter")
        
        # Skriv ut resultatene
        for i, doc in enumerate(docs, 1):
            print(f"\nDokument {i}:")
            print(f"Kilde: {doc.metadata.get('source', 'Ukjent')}")
            print(f"Innhold: {doc.page_content[:200]}...")
        
        return docs
    except Exception as e:
        logger.error(f"Feil ved søk i Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Pinecone-tilkobling og søk")
    parser.add_argument("--connection-only", action="store_true", help="Test bare tilkoblingen, ikke søk")
    parser.add_argument("--query", "-q", type=str, default="Hva sier loven om foreldrepermisjoner?", 
                        help="Spørringen som skal brukes for søk")
    parser.add_argument("--limit", "-k", type=int, default=5,
                        help="Antall dokumenter å hente")
    
    args = parser.parse_args()
    
    # Test tilkobling
    if not test_pinecone_connection():
        sys.exit(1)
    
    # Test søk hvis ikke bare tilkobling
    if not args.connection_only:
        print(f"\nUtfører søk med spørringen: '{args.query}'")
        docs = test_pinecone_search(args.query, args.limit)
        
        if not docs:
            print("Ingen dokumenter funnet eller feil oppsto under søket.")
            sys.exit(1)
        
        print(f"\nFant {len(docs)} dokumenter") 