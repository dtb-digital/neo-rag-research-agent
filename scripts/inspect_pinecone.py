#!/usr/bin/env python
"""
Inspiser dokument-strukturen i Pinecone-indeksen.
Dette skriptet gjør en enkel spørring mot Pinecone for å se på strukturen til dokumentene.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Last inn miljøvariabler fra .env-fil
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)

# Sjekk at nødvendige miljøvariabler er satt
required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
for var in required_vars:
    if not os.environ.get(var):
        raise EnvironmentError(f"Mangler miljøvariabel: {var}")

# Importer Pinecone
from pinecone import Pinecone, PodSpec, ServerlessSpec
from openai import OpenAI
import numpy as np

# Opprett Pinecone-klient
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX_NAME"]

# Opprett OpenAI-klient for embeddings
client = OpenAI()

# Funksjon for å hente embeddings fra OpenAI
def get_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Spørringstekster
query_texts = [
    "Hva er hovedprinsippene i offentlighetsloven?",
    "Offentlighetsloven åpenhet",
    "Innsyn i dokumenter offentlighetsloven"
]

# Hent indeks
try:
    index = pc.Index(index_name)
    print(f"Tilkoblet Pinecone-indeks: {index_name}")
    
    # Test spørringer
    for query in query_texts:
        print(f"\n\n--- Spørring: '{query}' ---")
        
        # Generer embedding
        query_embedding = get_embeddings(query)
        
        # Utfør søk
        results = index.query(
            vector=query_embedding,
            top_k=3,  # Hent 3 resultater
            include_metadata=True
        )
        
        # Skriv ut resultater
        print(f"Antall resultater: {len(results['matches'])}")
        
        for i, match in enumerate(results['matches']):
            print(f"\nResultat #{i+1} (score: {match['score']:.4f}):")
            print("Metadata struktur:")
            
            if match.get('metadata'):
                # Vis nøkler i metadata
                print(f"Metadata nøkler: {list(match['metadata'].keys())}")
                
                # Vis en kort del av teksten (hvis den finnes)
                for text_key in ['text', 'content', 'page_content', 'chunk', 'passage']:
                    if text_key in match['metadata']:
                        text_content = match['metadata'][text_key]
                        print(f"\nInnhold fra '{text_key}':")
                        print(text_content[:300] + "..." if len(text_content) > 300 else text_content)
                        break
            else:
                print("Ingen metadata funnet")
                
        print("\n" + "-"*50)

except Exception as e:
    print(f"Feil ved tilkobling til Pinecone: {e}") 