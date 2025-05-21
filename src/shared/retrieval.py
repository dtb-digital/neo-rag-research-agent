"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.
"""

import os
from contextlib import contextmanager
from typing import Generator
import logging
import json
import traceback

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import BaseConfiguration

# Sett opp en global logger for retrieval-modulen
logger = logging.getLogger("retrieval")

# Konstant for dokumentstruktur
TEXT_FIELD = "text"  # Vi standardiserer på dette feltet for dokumentinnhold

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    elif provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=model)  # type: ignore
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    # Standardiser på TEXT_FIELD
    yield vstore.as_retriever(
        search_kwargs=configuration.search_kwargs,
        content_key=TEXT_FIELD
    )


@contextmanager
def make_pinecone_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore
    import pinecone
    
    # Koble til Pinecone
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Bruk miljøvariabel hvis den finnes, ellers bruk standardverdien
    index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
    logger.info(f"Kobler til Pinecone indeks: {index_name}")
    
    # Logg miljøvariabler
    logger.info("Miljøvariabler:")
    logger.info(f"PINECONE_API_KEY: {'SATT' if os.environ.get('PINECONE_API_KEY') else 'IKKE SATT'}")
    logger.info(f"PINECONE_INDEX_NAME: {index_name}")
    logger.info(f"OPENAI_API_KEY: {'SATT' if os.environ.get('OPENAI_API_KEY') else 'IKKE SATT'}")
    
    try:
        # Opprett Pinecone Index og VectorStore
        logger.info(f"Oppretter Pinecone Index og PineconeVectorStore med text_key='text'")
        index = pc.Index(index_name)
        
        # Logg embedding model
        logger.info(f"Embedding modell type: {type(embedding_model)}")
        
        # Opprett VectorStore med korrekt text_key parameter
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="text"  # Dette er nøkkelen i Pinecone som inneholder dokumentteksten
        )
        
        # Logg typer for debugging
        logger.info(f"Pinecone Index type: {type(index)}")
        logger.info(f"VectorStore type: {type(vectorstore)}")
        
        # Hent et eksempeldokument for å verifisere struktur hvis mulig
        try:
            # Prøv å hente et dokument for å verifisere strukturen
            from pinecone import QueryFilter
            results = index.query(
                vector=[0.0] * 1536,  # Dummy vector
                top_k=1,
                include_metadata=True
            )
            if results.matches and len(results.matches) > 0:
                sample_doc = results.matches[0]
                logger.info(f"Eksempel på dokumentstruktur fra Pinecone:")
                logger.info(f"ID: {sample_doc.id}")
                logger.info(f"Metadata nøkler: {list(sample_doc.metadata.keys()) if sample_doc.metadata else 'Ingen metadata'}")
                if "text" in sample_doc.metadata:
                    logger.info("BEKREFTET: 'text' finnes i metadata")
                else:
                    logger.warning("ADVARSEL: 'text' finnes IKKE i metadata")
        except Exception as e:
            logger.warning(f"Kunne ikke hente eksempeldokument: {str(e)}")
        
        # Opprett retriever med korrekt content_key
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5},
            content_key="text"  # Forteller retrieveren å bruke 'text' feltet fra Pinecone
        )
        
        logger.info(f"Retriever opprettet: {type(retriever)}")
        
        try:
            yield retriever
        finally:
            # Gjør nødvendig cleanup her hvis det trengs
            pass
            
    except Exception as e:
        logger.error(f"Feil ved oppretting av Pinecone retriever: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise


@contextmanager
def make_mongodb_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    
    # Standardiser på TEXT_FIELD
    yield vstore.as_retriever(
        search_kwargs=configuration.search_kwargs,
        content_key=TEXT_FIELD
    )


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    
    provider = configuration.retriever_provider
    if provider == "elastic" or provider == "elastic-local":
        with make_elastic_retriever(configuration, embedding_model) as retriever:
            yield retriever
    elif provider == "pinecone":
        with make_pinecone_retriever(configuration, embedding_model) as retriever:
            yield retriever
    elif provider == "mongodb":
        with make_mongodb_retriever(configuration, embedding_model) as retriever:
            yield retriever
    else:
        raise ValueError(
            "Unrecognized retriever_provider in configuration. "
            f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
            f"Got: {configuration.retriever_provider}"
        )
