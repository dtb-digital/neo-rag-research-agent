import os
from contextlib import contextmanager
from typing import Generator

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langsmith import expect, unit

# Importere config-modulen for å sikre at .env-filen lastes
import src.config  # Dette vil laste miljøvariabler fra .env

from index_graph import graph as index_graph
from retrieval_graph import graph
from shared.configuration import BaseConfiguration
from shared.retrieval import make_text_encoder


@contextmanager
def make_elastic_vectorstore(
    configuration: BaseConfiguration,
) -> Generator[VectorStore, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    embedding_model = make_text_encoder(configuration.embedding_model)
    vstore = ElasticsearchStore(
        es_user=os.environ["ELASTICSEARCH_USER"],
        es_password=os.environ["ELASTICSEARCH_PASSWORD"],
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )
    yield vstore


@pytest.mark.asyncio
@unit
async def test_retrieval_graph() -> None:
    """Test at spørringsklassifisering fungerer riktig med hovedgrafen.
    
    Denne testen fokuserer kun på klassifisering av ulike spørringstyper
    og hopper over Elasticsearch-funksjonaliteten som ikke brukes i prosjektet.
    """
    # Konfigurer test med Pinecone som retriever
    config = RunnableConfig(
        configurable={
            "retriever_provider": "pinecone",
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "openai/gpt-4o-mini",
            "response_model": "openai/gpt-4o-mini",
        }
    )
    
    # Test generell spørring
    res = await graph.ainvoke(
        {"messages": [("user", "Hei! Hvordan går det?")]},
        config,
    )
    print(f"Klassifisering av generell spørring: {res['router']['type']}")
    expect(res["router"]["type"]).to_contain("generelt")

    # Test første lovrelatert spørring (lovspørsmål)
    res = await graph.ainvoke(
        {"messages": [("user", "Hva sier offentlighetsloven om hva som er unntatt offentlighet?")]},
        config,
    )
    print(f"Klassifisering av første lovspørsmål: {res['router']['type']}")
    expect(res["router"]["type"]).to_contain("lovspørsmål")
    
    # Test andre lovrelatert spørring (lovspørsmål)
    res = await graph.ainvoke(
        {"messages": [("user", "Hvilke hovedprinsipper inneholder offentlighetsloven?")]},
        config,
    )
    print(f"Klassifisering av andre lovspørsmål: {res['router']['type']}")
    expect(res["router"]["type"]).to_contain("lovspørsmål")

@pytest.mark.asyncio
async def test_offentlighetsloven_query() -> None:
    """Test hovedgrafen med en spørring om offentlighetsloven med Pinecone-retriever."""
    # Sett opp konfigurasjon med OpenAI-modeller
    config = RunnableConfig(
        configurable={
            "retriever_provider": "pinecone",  # Bruk Pinecone for å hente lovtekster
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "openai/gpt-4o-mini",     # Bruk gpt-4o-mini som støtter structured output
            "response_model": "openai/gpt-4o-mini",  # Bruk gpt-4o-mini for alle svar
        }
    )
    
    # Test spørring om offentlighetsloven
    res = await graph.ainvoke(
        {"messages": [("user", "Hva er hovedprinsippene i offentlighetsloven?")]},
        config,
    )
    
    # For denne testen forventer vi at spørringen blir klassifisert som "lovspørsmål"
    # siden vi nå bruker Pinecone-retrieveren som inneholder lovdata
    assert res["router"]["type"] == "lovspørsmål", "Spørringen ble ikke klassifisert som lovspørsmål"
    
    # Verifiser at svaret inneholder relevante begreper fra offentlighetsloven
    # basert på faktisk innhold i Pinecone-indeksen
    response = str(res["messages"][-1].content).lower()
    
    # Sjekk om svaret inneholder minst ett av disse begrepene
    relevant_terms = [
        "offentlighet",
        "innsyn",
        "offentlighetsprinsippet",
        "offentleg",
        "verksemd",
        "dokument"
    ]
    
    found_terms = [term for term in relevant_terms if term in response]
    assert found_terms, f"Svaret mangler relevante begreper fra offentlighetsloven. Forventet minst ett av: {relevant_terms}"
    
    # Skriv ut svaret for debugging
    print(f"Svar: {res['messages'][-1].content}")
