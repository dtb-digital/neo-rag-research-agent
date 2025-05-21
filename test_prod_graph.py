#!/usr/bin/env python3
"""
Test av hele grafen som i produksjonsmiljø uten tilpasninger.
Dette skal kjøre serveren akkurat slik den kjører i prod.
"""

import os
import sys
import json
import asyncio
import logging
import traceback
import dotenv
from contextlib import contextmanager

# Sett opp detaljert logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("prod_graph_test.log")
    ],
)
logger = logging.getLogger("test_prod_graph")

# Last inn miljøvariabler og sett opp Python-søkestien
print("Laster miljøvariabler fra .env")
dotenv.load_dotenv()

# Legg til src i søkestien
sys.path.insert(0, os.path.abspath("src"))
os.environ["PYTHONPATH"] = os.path.abspath("src")

# Importer nødvendige moduler etter at søkestien er satt opp
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

# Importer moduler fra prosjektet
from retrieval_graph.graph import graph as retrieval_graph
from retrieval_graph.state import AgentState, InputState
from retrieval_graph.graph import conduct_research as original_conduct_research
from retrieval_graph.graph import respond as original_respond
from shared.retrieval import make_retriever, make_pinecone_retriever, make_text_encoder

#################################################
# DEL 1: Logger-klasser og hjelpefunksjoner
#################################################

class DocumentLogger:
    @staticmethod
    def log_documents(docs, prefix=""):
        """Logger detaljer om dokumenter for debugging."""
        if not docs:
            logger.warning(f"{prefix} Ingen dokumenter funnet!")
            return
        
        logger.info(f"{prefix} Fant {len(docs)} dokumenter")
        for i, doc in enumerate(docs[:3]):  # Log de første 3 dokumentene
            logger.info(f"{prefix} Dokument {i}:")
            logger.info(f"  Type: {type(doc)}")
            
            # Sjekk page_content
            if hasattr(doc, "page_content"):
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.info(f"  page_content: {content_preview}")
            else:
                logger.warning(f"  Dokument mangler page_content attributt")
            
            # Sjekk text-feltet
            if hasattr(doc, "text"):
                text_preview = doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
                logger.info(f"  text: {text_preview}")
            else:
                logger.warning(f"  Dokument mangler text-attributt")
            
            # Sjekk metadata
            if hasattr(doc, "metadata"):
                metadata_keys = list(doc.metadata.keys())
                logger.info(f"  metadata keys: {metadata_keys}")
                # Logg viktige metadata-felter
                important_fields = ["lov_id", "lov_navn", "paragraf_nr", "kapittel_nr"]
                for field in important_fields:
                    if field in doc.metadata:
                        logger.info(f"  metadata.{field}: {doc.metadata[field]}")
            else:
                logger.warning(f"  Dokument mangler metadata-attributt")
            
            # Logg alle attributter for å se om noe mangler
            logger.info(f"  Alle attributter: {dir(doc)}")

#################################################
# DEL 2: Monkey-patching av moduler
#################################################

# -------------------------------
# make_retriever og make_pinecone_retriever
# -------------------------------
original_make_retriever = make_retriever
original_make_pinecone_retriever = make_pinecone_retriever

def logged_make_retriever(config):
    """Wrapper for make_retriever som legger til logging."""
    logger.info("=== make_retriever kalt ===")
    logger.info(f"Konfigurasjon: {config}")
    
    try:
        # Kjør den originale funksjonen
        retriever = original_make_retriever(config)
        
        # Logg retriever detaljer
        logger.info(f"Retriever type: {type(retriever)}")
        logger.info(f"Retriever egenskaper: {dir(retriever)}")
        
        # Prøv å inspisere vector store
        vs = retriever.vectorstore if hasattr(retriever, "vectorstore") else None
        if vs:
            logger.info(f"VectorStore type: {type(vs)}")
            logger.info(f"VectorStore egenskaper: {dir(vs)}")
            
            # Sjekk om vi har text_key og content_key
            if hasattr(vs, "text_key"):
                logger.info(f"VectorStore text_key: {vs.text_key}")
            else:
                logger.warning("VectorStore mangler text_key attributt")
                
            if hasattr(vs, "content_key"):
                logger.info(f"VectorStore content_key: {vs.content_key}")
            else:
                logger.warning("VectorStore mangler content_key attributt")
            
            # Se på index_name
            if hasattr(vs, "index_name"):
                logger.info(f"VectorStore index_name: {vs.index_name}")
        else:
            logger.warning("Kunne ikke finne vectorstore attributt på retrieveren")
        
        return retriever
    except Exception as e:
        logger.error(f"Feil i make_retriever: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@contextmanager
def logged_make_pinecone_retriever(configuration, embedding_model):
    """Wrapper for make_pinecone_retriever som legger til logging."""
    logger.info("=== make_pinecone_retriever kalt ===")
    logger.info(f"Konfigurasjon type: {type(configuration)}")
    
    # Dump viktige konfigurasjonsparametere
    try:
        if hasattr(configuration, "pinecone_index"):
            logger.info(f"configuration.pinecone_index: {configuration.pinecone_index}")
        if hasattr(configuration, "pinecone_api_key"):
            logger.info(f"configuration.pinecone_api_key finnes: {'ja' if configuration.pinecone_api_key else 'nei'}")
    except Exception as e:
        logger.error(f"Feil ved logging av konfigurasjon: {str(e)}")
    
    logger.info(f"Embedding model type: {type(embedding_model)}")
    
    # Undersøk call stacken for å se hvor denne blir kalt fra
    stack = traceback.extract_stack()
    logger.info(f"Kalt fra: {stack[-2].filename}:{stack[-2].lineno}")
    
    try:
        # Kjør den originale funksjonen
        with original_make_pinecone_retriever(configuration, embedding_model) as retriever:
            # Logg retriever informasjon
            logger.info(f"make_pinecone_retriever ga retriever av type: {type(retriever)}")
            
            # Se på vector store
            if hasattr(retriever, "vectorstore"):
                vs = retriever.vectorstore
                logger.info(f"VectorStore type: {type(vs)}")
                
                # Sjekk text_key og content_key
                if hasattr(vs, "text_key"):
                    logger.info(f"VectorStore text_key: {vs.text_key}")
                else:
                    logger.warning("VectorStore mangler text_key attributt")
                
                if hasattr(vs, "content_key"):
                    logger.info(f"VectorStore content_key: {vs.content_key}")
                else:
                    logger.warning("VectorStore mangler content_key attributt")
            else:
                logger.warning("Kunne ikke finne vectorstore på retriever")
            
            # Yield retrieveren til kalleren
            yield retriever
    except Exception as e:
        logger.error(f"Feil i make_pinecone_retriever: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# -------------------------------
# PineconeVectorStore klassen
# -------------------------------
original_similarity_search = PineconeVectorStore.similarity_search
original_from_existing_index = PineconeVectorStore.from_existing_index

@classmethod
def logged_from_existing_index(cls, index, embedding, text_key="text", namespace=None, **kwargs):
    """Wrapper for from_existing_index med logging."""
    logger.info("=== PineconeVectorStore.from_existing_index kalt ===")
    logger.info(f"Index type: {type(index)}")
    logger.info(f"Embedding type: {type(embedding)}")
    logger.info(f"text_key: {text_key}")
    logger.info(f"namespace: {namespace}")
    logger.info(f"kwargs: {kwargs}")
    
    # Undersøk call stacken for å se hvor denne blir kalt fra
    stack = traceback.extract_stack()
    logger.info(f"Kalt fra: {stack[-2].filename}:{stack[-2].lineno}")
    
    try:
        # Kjør den originale funksjonen
        vectorstore = original_from_existing_index(index, embedding, text_key, namespace, **kwargs)
        
        # Logg resultater
        logger.info(f"Opprettet PineconeVectorStore med text_key={getattr(vectorstore, 'text_key', 'None')}")
        logger.info(f"Opprettet PineconeVectorStore med metadata_key={getattr(vectorstore, 'metadata_key', 'None')}")
        if hasattr(vectorstore, 'content_key'):
            logger.info(f"Opprettet PineconeVectorStore med content_key={vectorstore.content_key}")
        else:
            logger.warning("Opprettet PineconeVectorStore MANGLER content_key")
        
        return vectorstore
    except Exception as e:
        logger.error(f"Feil i from_existing_index: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def logged_similarity_search(self, query, k=4, filter=None, namespace=None, **kwargs):
    """Wrapper for similarity_search med logging."""
    logger.info("=== PineconeVectorStore.similarity_search kalt ===")
    logger.info(f"Spørring: '{query}', k={k}, filter={filter}, namespace={namespace}, kwargs={kwargs}")
    logger.info(f"PineconeVectorStore konfigurert med: text_key={getattr(self, 'text_key', 'None')}, embedding={type(getattr(self, '_embedding', None))}")
    
    try:
        # Kjør den originale funksjonen
        docs = original_similarity_search(self, query, k, filter, namespace, **kwargs)
        
        # Logg resultatene
        logger.info(f"similarity_search returnerte {len(docs)} dokumenter")
        for i, doc in enumerate(docs[:2]):  # Bare logg de to første for oversikt
            logger.info(f"Dokument {i} type: {type(doc)}")
            if hasattr(doc, "page_content"):
                logger.info(f"Dokument {i} har page_content ({len(doc.page_content)} tegn)")
            if hasattr(doc, "text"):
                logger.info(f"Dokument {i} har text-felt ({len(doc.text)} tegn)")
            if hasattr(doc, "metadata"):
                logger.info(f"Dokument {i} har metadata: {list(doc.metadata.keys())}")
        
        return docs
    except Exception as e:
        logger.error(f"Feil i similarity_search: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# -------------------------------
# Document-klassen
# -------------------------------
original_document_init = Document.__init__
original_document_getattribute = Document.__getattribute__

def logged_document_init(self, page_content, metadata=None, **kwargs):
    """Wrapper for Document.__init__ med logging."""
    # Logg kun noen dokumenter for å begrense mengden output
    if not hasattr(Document, "_log_counter"):
        Document._log_counter = 0
    
    # Logg hver 10. dokument-opprettelse for å begrense mengden
    if Document._log_counter % 10 == 0:
        logger.info("=== Document.__init__ kalt ===")
        
        # Logg call stack for å se hvor dokumentet ble opprettet
        stack = traceback.extract_stack()
        caller = stack[-2]
        logger.info(f"Dokument opprettet fra: {caller.filename}:{caller.lineno}")
        
        # Logg dokumentets innhold og feltene som settes
        content_preview = page_content[:100] + "..." if len(page_content) > 100 else page_content
        logger.info(f"page_content: {content_preview}")
        
        if metadata:
            logger.info(f"metadata nøkler: {list(metadata.keys())}")
            # Vis spesielt viktige metadata-felter
            for key in ["text", "lov_id", "lov_navn", "paragraf_nr"]:
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    logger.info(f"metadata['{key}']: {value}")
        
        for key, value in kwargs.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            logger.info(f"kwargs['{key}']: {value}")
    
    Document._log_counter += 1
    
    # Kall den originale metoden
    original_document_init(self, page_content, metadata, **kwargs)

def logged_document_getattribute(self, name):
    """Logger kritiske attributt-oppslag."""
    # Kun logg noen tilfeller for å begrense mengden output
    if name in ["text", "page_content"] and not hasattr(Document, "_getattr_counter"):
        Document._getattr_counter = 0
    
    if name in ["text", "page_content"] and Document._getattr_counter % 50 == 0:
        stack = traceback.extract_stack()
        caller = stack[-2]
        logger.info(f"Document.__getattribute__('{name}') kalt fra {caller.filename}:{caller.lineno}")
    
    if name in ["text", "page_content"]:
        Document._getattr_counter = getattr(Document, "_getattr_counter", 0) + 1
    
    return original_document_getattribute(self, name)

# -------------------------------
# conduct_research og respond
# -------------------------------
async def logged_conduct_research(state, *, config):
    """Wrapper for conduct_research som legger til logging."""
    logger.info("=== conduct_research starter ===")
    logger.info(f"State før conduct_research: messages={len(state.messages)}, steps={state.steps}")
    
    if hasattr(state, "documents"):
        DocumentLogger.log_documents(state.documents, "State før conduct_research")
    
    try:
        # Kjør den originale funksjonen
        result = await original_conduct_research(state, config=config)
        
        # Logg resultatet
        logger.info("=== conduct_research ferdig ===")
        if "documents" in result:
            DocumentLogger.log_documents(result["documents"], "conduct_research resultat")
        else:
            logger.warning("conduct_research returnerte ingen dokumenter i resultat")
        
        logger.info(f"Gjenværende steg: {result.get('steps', [])}")
        return result
    except Exception as e:
        logger.error(f"Feil i conduct_research: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def logged_respond(state, *, config):
    """Wrapper for respond som legger til logging."""
    logger.info("=== respond starter ===")
    logger.info(f"State før respond: messages={len(state.messages)}")
    
    if hasattr(state, "documents"):
        DocumentLogger.log_documents(state.documents, "State før respond")
    
    try:
        # Kjør den originale funksjonen
        result = await original_respond(state, config=config)
        
        # Logg resultatet
        logger.info("=== respond ferdig ===")
        if "messages" in result:
            logger.info(f"Antall meldinger i resultat: {len(result['messages'])}")
            if result["messages"]:
                content_preview = result["messages"][0].content[:200] + "..." if len(result["messages"][0].content) > 200 else result["messages"][0].content
                logger.info(f"Første melding innhold: {content_preview}")
        else:
            logger.warning("respond returnerte ingen meldinger i resultat")
        
        return result
    except Exception as e:
        logger.error(f"Feil i respond: {str(e)}")
        logger.error(traceback.format_exc())
        raise

#################################################
# DEL 3: Monkeypatching av moduler
#################################################

# Erstatt de originale funksjonene med våre loggede versjoner
import shared.retrieval
import retrieval_graph.graph

# Patch retrieval-relaterte funksjoner
shared.retrieval.make_retriever = logged_make_retriever
shared.retrieval.make_pinecone_retriever = logged_make_pinecone_retriever
PineconeVectorStore.similarity_search = logged_similarity_search
PineconeVectorStore.from_existing_index = logged_from_existing_index

# Patch Document-klassen
Document.__init__ = logged_document_init
Document.__getattribute__ = logged_document_getattribute

# Patch graf-funksjoner
retrieval_graph.graph.conduct_research = logged_conduct_research
retrieval_graph.graph.respond = logged_respond

#################################################
# DEL 4: Hovedfunksjon
#################################################

async def main():
    """Hovedfunksjon som kjører testen."""
    logger.info("======== START TEST_PROD_GRAPH ========")
    
    try:
        # Sjekk og verifiser miljøvariabler
        pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
        
        logger.info(f"Miljøvariabler: PINECONE_API_KEY {'funnet' if pinecone_api_key else 'MANGLER'}")
        logger.info(f"Miljøvariabler: OPENAI_API_KEY {'funnet' if openai_api_key else 'MANGLER'}")
        logger.info(f"Bruker Pinecone indeks: {pinecone_index_name}")
        
        if not pinecone_api_key or not openai_api_key:
            logger.error("VIKTIG: En eller flere nødvendige miljøvariabler mangler!")
            return
        
        # Spørring som skal testes
        query = "hva sier offentlighetsloven om innsynskrav"
        logger.info(f"Testing spørring: '{query}'")
        
        # Opprett input state med spørringen
        input_state = InputState(messages=[HumanMessage(content=query)])
        
        # Kjør grafen
        logger.info("Kjører retrieval_graph...")
        result = await retrieval_graph.ainvoke(input_state)
        
        # Logg resultatet
        logger.info("Grafen har returnert")
        if "messages" in result:
            logger.info(f"Antall meldinger i sluttresultat: {len(result['messages'])}")
            if result["messages"]:
                content_preview = result["messages"][-1].content[:200] + "..." if len(result["messages"][-1].content) > 200 else result["messages"][-1].content
                logger.info(f"Siste melding innhold: {content_preview}")
        else:
            logger.warning("Grafen returnerte ingen meldinger i sluttresultat")
        
        if "documents" in result:
            DocumentLogger.log_documents(result["documents"], "Sluttresultat")
        else:
            logger.info("Ingen documents-nøkkel i sluttresultatet (dette er normalt for fullført graf)")
        
        logger.info("TEST FULLFØRT")
        
    except Exception as e:
        logger.error(f"TEST FEILET: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("======== SLUTT TEST_PROD_GRAPH ========")

if __name__ == "__main__":
    asyncio.run(main()) 