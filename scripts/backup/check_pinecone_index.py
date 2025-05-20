#!/usr/bin/env python
"""
Sjekk innholdet i Pinecone-indeksen direkte for å finne ut hva som er lagret der.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
import pinecone

# Last inn miljøvariabler fra .env-filen
load_dotenv()

# Oppsett av logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("check-pinecone")

def main():
    parser = argparse.ArgumentParser(description="Sjekk innholdet i Pinecone-indeksen")
    parser.add_argument("--limit", type=int, default=5, help="Antall vektorer å vise")
    parser.add_argument("--index", help="Pinecone-indeksnavn (standard: fra miljøvariabel)")
    parser.add_argument("--environment", help="Pinecone-miljø (standard: fra miljøvariabel)")
    parser.add_argument("--host", help="Pinecone-vertsnavn (standard: fra miljøvariabel)")
    parser.add_argument("--ids", nargs='+', help="Spesifikke vektor-ID-er å sjekke")
    
    args = parser.parse_args()
    
    # Hent Pinecone-konfigurasjon fra miljøvariabler eller parametre
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = args.index or os.environ.get("PINECONE_INDEX_NAME", "lovdata-index")
    environment = args.environment or os.environ.get("PINECONE_ENVIRONMENT", "us-east-1-aws")
    host = args.host or os.environ.get("PINECONE_HOST")
    
    if not api_key:
        logger.error("PINECONE_API_KEY er ikke satt. Kan ikke fortsette.")
        return
    
    logger.info(f"Kobler til Pinecone med følgende konfigurasjon:")
    logger.info(f"Index: {index_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Host: {host or '(bruk standard)'}")
    
    # Initialiser Pinecone-klienten
    try:
        pinecone.init(api_key=api_key, environment=environment)
        logger.info(f"Pinecone-klient initialisert")
        
        # Vis alle tilgjengelige indekser
        logger.info(f"Tilgjengelige indekser: {pinecone.list_indexes()}")
        
        # Koble til indeksen
        index = pinecone.Index(index_name)
        logger.info(f"Koblet til indeks: {index_name}")
        
        # Hent statistikk om indeksen
        stats = index.describe_index_stats()
        logger.info(f"Indeksstatistikk: {stats}")
        
        # Hent og vis vektorer
        if args.ids:
            # Hent spesifikke vektorer etter ID
            logger.info(f"Henter vektorer med ID-er: {args.ids}")
            vectors = index.fetch(ids=args.ids)
            
            logger.info(f"Fant {len(vectors.vectors)} vektorer:")
            for id, vector in vectors.vectors.items():
                logger.info(f"\nID: {id}")
                metadata = vector.get('metadata', {})
                logger.info(f"Metadata: {metadata}")
        else:
            # Vis eksempelspørring
            logger.info(f"Utfører eksempelspørring for å vise vektorer...")
            
            # Lager en tom vektor av riktig dimensjon
            dim = stats.get("dimension", 1536)  # Standard er 1536 for OpenAI embeddings
            dummy_vector = [0.0] * dim
            
            # Kjør spørring for å hente noen vektorer
            query_response = index.query(
                vector=dummy_vector,
                top_k=args.limit,
                include_metadata=True
            )
            
            # Vis resultater
            matches = query_response.get("matches", [])
            logger.info(f"Fant {len(matches)} vektorer med spørringen:")
            
            for i, match in enumerate(matches):
                logger.info(f"\nResultat {i+1}:")
                logger.info(f"ID: {match.get('id', 'ukjent')}")
                logger.info(f"Score: {match.get('score', 'ukjent')}")
                
                # Vis metadata
                metadata = match.get("metadata", {})
                logger.info(f"Metadata nøkler: {list(metadata.keys())}")
                
                # Sjekk om tekst-feltet finnes og vis det
                if "text" in metadata:
                    text_preview = metadata["text"][:300] + "..." if len(metadata["text"]) > 300 else metadata["text"]
                    logger.info(f"Text: {text_preview}")
                elif "page_content" in metadata:
                    text_preview = metadata["page_content"][:300] + "..." if len(metadata["page_content"]) > 300 else metadata["page_content"]
                    logger.info(f"Page Content: {text_preview}")
                else:
                    logger.info(f"Ingen tekstfelt funnet i metadata")
                    
                # Vis all metadata for første resultat
                if i == 0:
                    logger.info(f"Komplett metadata for første resultat:")
                    for key, value in metadata.items():
                        if isinstance(value, str) and len(value) > 100:
                            logger.info(f"  {key}: {value[:100]}...")
                        else:
                            logger.info(f"  {key}: {value}")
    
    except Exception as e:
        logger.exception(f"Feil under tilkobling til Pinecone: {e}")
    finally:
        # Rydd opp
        pinecone.deinit()

if __name__ == "__main__":
    main() 