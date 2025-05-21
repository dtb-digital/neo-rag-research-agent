"""Test direkte retrieval fra Pinecone og conduct_research funksjonen."""

import os
import sys
import json
import asyncio
import logging
import traceback
import dotenv

# Last inn miljøvariabler fra .env-filen
print("Laster miljøvariabler fra .env")
dotenv.load_dotenv()

# Riktig oppsett av Python-søkestien
sys.path.insert(0, os.path.abspath("src"))
os.environ["PYTHONPATH"] = os.path.abspath("src")

# Sett opp logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("debug_tests.log")],
)
logger = logging.getLogger("test_graph_debug")

# Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableConfig

# Prosjekt-imports - nå uten src. prefiks siden vi har lagt src til i Python-søkestien
from retrieval_graph.graph import conduct_research  
from shared.state import reduce_docs
from retrieval_graph.state import AgentState, Router

async def main():
    """Kjør tester for Pinecone-retrieval og conduct_research."""
    logger.info("=== START TEST_GRAPH_DEBUG ===")
    
    # Sjekk og verifiser at alle nødvendige miljøvariabler er lastet
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
    
    logger.info(f"Miljøvariabler lastet: PINECONE_API_KEY {'funnet' if pinecone_api_key else 'MANGLER'}")
    logger.info(f"Miljøvariabler lastet: OPENAI_API_KEY {'funnet' if openai_api_key else 'MANGLER'}")
    logger.info(f"Bruker Pinecone indeks: {pinecone_index_name}")
    
    if not pinecone_api_key or not openai_api_key:
        logger.error("VIKTIG: En eller flere nødvendige miljøvariabler mangler!")
        logger.error("Sørg for at .env-filen eksisterer og inneholder PINECONE_API_KEY og OPENAI_API_KEY")
        return
    
    # Opprett embedding-modell
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Test retrieval direkte fra Pinecone først for å isolere problemer
    logger.info("=== TESTER DIREKTE PINECONE RETRIEVAL ===")
    try:
        # Koble til Pinecone
        pc = pinecone.Pinecone(api_key=pinecone_api_key)
        index = pc.Index(name=pinecone_index_name)
        
        # Opprett vektorlager
        vstore = PineconeVectorStore.from_existing_index(
            index_name=pinecone_index_name,
            embedding=embedding_model,
            text_key="text"  # Bruk text for nye dokumenter i den nye indeksen
        )
        
        # Opprett retriever med content_key="text"
        retriever = vstore.as_retriever(
            search_kwargs={"k": 5},
            content_key="text"  # Nøkkelen som inneholder dokumentteksten
        )
        
        # Test søk
        test_query = "Hva er reglene for habilitet?"
        logger.info(f"Tester søk med spørring: '{test_query}'")
        
        # Embed spørringen
        query_embedding = embedding_model.embed_query(test_query)
        logger.info(f"Embedding-dimensjon: {len(query_embedding)}")
        
        # Utfør direkte søk
        test_docs = await retriever.ainvoke(test_query)
        logger.info(f"Retriever fant {len(test_docs)} dokumenter")
        
        # Skriv ut detaljer om dokumentene
        if test_docs:
            logger.info(f"Første dokument metadata: {test_docs[0].metadata}")
            logger.info(f"Første dokument type: {type(test_docs[0])}")
            logger.info(f"Første dokument innhold (utdrag): {test_docs[0].page_content[:100]}")
        else:
            logger.warning("Ingen dokumenter funnet ved direkte Pinecone-søk!")
    except Exception as e:
        logger.error(f"Feil ved testing av direkte Pinecone retrieval: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Test conduct_research
    logger.info("=== TESTER CONDUCT_RESEARCH ===")
    try:
        # Spørsmål
        query = "Hva er reglene for habilitet?"
        
        # Opprett LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Oppsett av konfigurasjon
        config = RunnableConfig(
            configurable={
                "retriever_provider": "pinecone",
                "embedding_model": "openai/text-embedding-3-small",
                "query_model": "openai/gpt-4o",
                "response_model": "openai/gpt-4o",
                "search_kwargs": {"k": 10}
            }
        )
        
        # Opprett initial_state med tom docs-liste
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            router=Router(type="lovspørsmål", logic=""),
            steps=["Undersøk lovregler om habilitet"],
            documents=[]
        )
        
        logger.info(f"Spørring: '{query}'")
        
        # Kall conduct_research og gi den state og config
        result = await conduct_research(initial_state, config=config)
        
        # Sjekk resultatet - dokumentene vil være i result['documents']
        if 'documents' in result and result['documents']:
            documents = result['documents']
            logger.info(f"Totalt antall dokumenter før reduksjon: {len(documents)}")
            
            # Test reduksjon av dokumenter
            unique_docs = reduce_docs([], documents)
            logger.info(f"Antall unike dokumenter etter reduksjon: {len(unique_docs)}")
            
            # Se på metadata
            if unique_docs:
                logger.info(f"Metadata nøkler fra første dokument: {list(unique_docs[0].metadata.keys())}")
                logger.info(f"Dokumentinnhold (første 100 tegn): {unique_docs[0].page_content[:100]}")
                logger.info(f"Unik ID: {unique_docs[0].metadata.get('uuid', 'ingen')}")
            else:
                logger.warning("Ingen dokumenter etter reduksjon!")
        else:
            logger.warning("Ingen dokumenter funnet i resultatet fra conduct_research!")
        
        # Print andre detaljer i resultatet
        logger.info("\nAndre detaljer i resultatet:")
        for key, value in result.items():
            if key != 'documents':
                logger.info(f"{key}: {str(value)[:200]}...")
        
    except Exception as e:
        logger.error(f"Feil under testing av conduct_research: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("=== SLUTT TEST_GRAPH_DEBUG ===")

if __name__ == "__main__":
    asyncio.run(main()) 