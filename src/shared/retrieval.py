"""Manage the Pinecone retriever configuration."""

import os
from contextlib import contextmanager
from typing import Generator
import logging
import traceback

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
import pinecone

from shared.configuration import BaseConfiguration

# Setup logging
logger = logging.getLogger("retrieval")

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

@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a Pinecone retriever based on the configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    
    # Connect to Pinecone
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
    
    try:
        # Create Pinecone Index and VectorStore
        index = pc.Index(index_name)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="text"
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5},
            content_key="text"
        )
        
        try:
            yield retriever
        finally:
            pass
            
    except Exception as e:
        logger.error(f"Error creating Pinecone retriever: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
