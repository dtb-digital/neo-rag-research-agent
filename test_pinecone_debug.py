"""Test direkte mot Pinecone for å debug'e problemer med dokumentstrukturen."""

import os
import asyncio
import sys
import dotenv
from pprint import pprint
from typing import List, Dict, Any

# Legg til src i Python-søkestien
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Last miljøvariabler
dotenv.load_dotenv()

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone

async def test_direct_pinecone():
    """Test direkte mot Pinecone API for å verifisere dokumentstruktur."""
    print("=== TEST DIREKTE MOT PINECONE ===")
    
    # Opprett embedding-modell
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("OpenAI embedding-modell opprettet")
    
    # Koble til Pinecone
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Velg indeksen
    index_name = os.environ["PINECONE_INDEX_NAME"]
    print(f"Kobler til indeks: {index_name}")
    
    # To ulike måter å opprette PineconeVectorStore på
    
    # 1. Bruk from_existing_index (anbefalt)
    print("\n-- Test med from_existing_index --")
    vstore1 = PineconeVectorStore.from_existing_index(
        index_name, embedding=embedding_model
    )
    
    # 2. Bruk direkte konstruktør (eldre)
    print("\n-- Test med direkte konstruktør --")
    index = pc.Index(name=index_name)
    vstore2 = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="text"  # Bruk "text" for å matche feltnavnet i Pinecone
    )
    
    # Test begge retrieverne
    query = "formål offentlighetsloven"
    
    # Test 1: med from_existing_index
    print(f"\nSøker med retriever1 (from_existing_index): '{query}'")
    retriever1 = vstore1.as_retriever(
        search_kwargs={"k": 5},
        content_key="text"  # Spesifiser content_key som "text" for å matche eksisterende dokument feltnavn
    )
    
    try:
        print("Retriever1 attributter:")
        print(f"  Klasse: {retriever1.__class__.__name__}")
        print(f"  Content key: {getattr(retriever1, 'content_key', 'Ukjent')}")
        
        # Lag embedding for spørringen
        query_embedding = embedding_model.embed_query(query)
        print(f"Query embedding dimensjon: {len(query_embedding)}")
        
        # Kjør søk
        docs1 = await retriever1.ainvoke(query)
        print(f"Søk fullført, fant {len(docs1)} dokumenter")
        
        if docs1:
            print("\nFørste dokument fra retriever1:")
            print(f"  Type: {type(docs1[0])}")
            print(f"  Metadata nøkler: {list(docs1[0].metadata.keys())}")
            print(f"  Har page_content: {'page_content' in dir(docs1[0])}")
            print(f"  Innhold (utdrag): {docs1[0].page_content[:100] if hasattr(docs1[0], 'page_content') else 'Ingen page_content funnet'}...")
            
            # Legg til ekstra debug for å se dokumentets attributter
            print("\nDocument Attributter:")
            doc = docs1[0]
            print(f"  Dir(document): {dir(doc)}")
            # Sjekk om dokumentet har riktige felt for å vises i Langsmith
            print(f"  Document.__dict__: {doc.__dict__}")
    except Exception as e:
        print(f"Feil ved søk med retriever1: {str(e)}")
    
    # Test 2: med direkte konstruktør
    print(f"\nSøker med retriever2 (direkte konstruktør): '{query}'")
    retriever2 = vstore2.as_retriever(
        search_kwargs={"k": 5}
    )
    
    try:
        print("Retriever2 attributter:")
        print(f"  Klasse: {retriever2.__class__.__name__}")
        print(f"  Content key: {getattr(retriever2, 'content_key', 'Ukjent')}")
        
        # Kjør søk
        docs2 = await retriever2.ainvoke(query)
        print(f"Søk fullført, fant {len(docs2)} dokumenter")
        
        if docs2:
            print("\nFørste dokument fra retriever2:")
            print(f"  Type: {type(docs2[0])}")
            print(f"  Metadata nøkler: {list(docs2[0].metadata.keys())}")
            print(f"  Har page_content: {'page_content' in dir(docs2[0])}")
            print(f"  Innhold (utdrag): {docs2[0].page_content[:100] if hasattr(docs2[0], 'page_content') else 'Ingen page_content funnet'}...")
    except Exception as e:
        print(f"Feil ved søk med retriever2: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_direct_pinecone()) 