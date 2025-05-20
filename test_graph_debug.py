import os
import sys
import asyncio
from typing import List, Dict, Any
from pprint import pprint

# Legg til src i Python-søkestien
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from retrieval_graph import graph
from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.state import AgentState, Router
from shared.retrieval import make_retriever
from shared.state import reduce_docs

# Sett miljøvariabler hvis nødvendig
import dotenv
dotenv.load_dotenv()

async def test_router():
    """Test at routeren klassifiserer spørsmålet riktig"""
    print("\n=== TESTER ROUTER ===")
    
    config = RunnableConfig(
        configurable={
            "retriever_provider": "pinecone",
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "openai/gpt-4o-mini",
            "response_model": "openai/gpt-4o-mini",
        }
    )
    
    query = "Hva er formålet med offentlighetsloven?"
    print(f"Spørring: {query}")
    
    # Kaller analyze_and_route_query direkte
    from retrieval_graph.graph import analyze_and_route_query
    
    # Opprett en enkel state
    state = AgentState(messages=[HumanMessage(content=query)])
    
    # Kjør routing
    result = await analyze_and_route_query(state, config=config)
    
    # Sjekk resultat
    print(f"Router resultat: {result}")
    return result

async def test_research_plan():
    """Test at forskningsplanen genereres riktig"""
    print("\n=== TESTER FORSKNINGSPLAN ===")
    
    config = RunnableConfig(
        configurable={
            "retriever_provider": "pinecone",
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "openai/gpt-4o-mini",
            "response_model": "openai/gpt-4o-mini",
        }
    )
    
    query = "Hva er formålet med offentlighetsloven?"
    print(f"Spørring: {query}")
    
    # Kaller create_research_plan direkte
    from retrieval_graph.graph import create_research_plan
    
    # Opprett en state med router-resultat
    state = AgentState(
        messages=[HumanMessage(content=query)],
        router=Router(type="lovspørsmål", logic="")
    )
    
    # Kjør plan-generering
    result = await create_research_plan(state, config=config)
    
    # Sjekk resultat
    print(f"Forskningsplan: {result}")
    return result

async def test_query_generation():
    """Test at spørringer genereres riktig basert på forskningsplanen"""
    print("\n=== TESTER QUERY-GENERERING ===")
    
    config = RunnableConfig(
        configurable={
            "retriever_provider": "pinecone",
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "openai/gpt-4o-mini",
            "response_model": "openai/gpt-4o-mini",
        }
    )
    
    query = "Hva er formålet med offentlighetsloven?"
    research_step = "Undersøk formålet med offentlighetsloven"
    print(f"Forskningssteg: {research_step}")
    
    # Kode fra conduct_research-funksjonen
    from retrieval_graph.researcher_graph.state import ResearcherState
    from retrieval_graph.graph import AgentConfiguration
    
    # Initialiser researcher state med spørsmålet
    researcher_state = ResearcherState(question=research_step)
    
    # Opprett konfigurasjonen
    configuration = AgentConfiguration.from_runnable_config(config)
    
    # Last modell
    from shared.utils import load_chat_model
    model = load_chat_model(configuration.query_model)
    
    # Definer respons-typen
    from typing import TypedDict
    class Response(TypedDict):
        queries: List[str]
    
    # Generer spørringer
    structured_model = model.with_structured_output(Response)
    messages = [
        {"role": "system", "content": configuration.generate_queries_system_prompt},
        {"role": "human", "content": researcher_state.question},
    ]
    
    response = await structured_model.ainvoke(messages)
    queries = response["queries"]
    
    # Sjekk resultat
    print(f"Genererte spørringer: {queries}")
    return queries

async def test_retrieval():
    """Test retrieval direkte med genererte spørringer"""
    print("\n=== TESTER RETRIEVAL ===")
    
    config = RunnableConfig(
        configurable={
            "retriever_provider": "pinecone",
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "openai/gpt-4o-mini",
            "response_model": "openai/gpt-4o-mini",
            "search_kwargs": {"k": 5}
        }
    )
    
    # Test både genererte og hardkodede spørringer
    queries = [
        "formål offentlighetsloven", 
        "offentlighetsloven paragraf 1",
        "offentlighet innsyn hovedprinsipper",
        "lov offentlighet åpenhet forvaltning"
    ]
    
    all_docs = []
    print(f"Kjører retrieval for {len(queries)} spørringer...")
    
    # Kjør retrieval for hver spørring
    with make_retriever(config) as retriever:
        for i, query in enumerate(queries):
            print(f"\nSpørring {i+1}: '{query}'")
            
            # Debug: Skriv ut retriever-informasjon
            print(f"Retriever type: {type(retriever)}")
            print(f"Retriever content_key: {getattr(retriever, 'content_key', 'Ukjent')}")
            
            # Gjør søk
            docs = await retriever.ainvoke(query, config)
            
            if docs:
                print(f"✅ Fant {len(docs)} dokumenter!")
                # Vis metadata for første dokument
                if len(docs) > 0:
                    print(f"Første dokument metadata: {docs[0].metadata}")
                    print(f"Første dokument innhold (utdrag): {docs[0].page_content[:100]}...")
                all_docs.extend(docs)
            else:
                print(f"❌ Ingen dokumenter funnet for '{query}'")
    
    # Sjekk reduksjon av duplikater
    print(f"\nTotalt antall dokumenter før reduksjon: {len(all_docs)}")
    unique_docs = reduce_docs(None, all_docs)
    print(f"Antall unike dokumenter etter reduksjon: {len(unique_docs)}")
    
    return unique_docs

async def test_full_research():
    """Test hele conduct_research-funksjonen"""
    print("\n=== TESTER FULL FORSKNINGSPROSESS ===")
    
    config = RunnableConfig(
        configurable={
            "retriever_provider": "pinecone",
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "openai/gpt-4o-mini",
            "response_model": "openai/gpt-4o-mini",
            "search_kwargs": {"k": 5}
        }
    )
    
    query = "Hva er formålet med offentlighetsloven?"
    research_step = "Undersøk formålet med offentlighetsloven"
    
    # Kaller conduct_research direkte
    from retrieval_graph.graph import conduct_research
    
    # Opprett en state med steps
    state = AgentState(
        messages=[HumanMessage(content=query)],
        router=Router(type="lovspørsmål", logic=""),
        steps=[research_step]
    )
    
    print(f"Kjører conduct_research med state:\n{state}")
    
    # Kjør forskning
    result = await conduct_research(state, config=config)
    
    # Sjekk resultat
    print(f"Forskningstresultat:\nNye dokumenter: {len(result.get('documents', []))}")
    print(f"Gjenværende steg: {result.get('steps', [])}")
    
    # Vis dokumentutdrag hvis noen ble funnet
    if result.get('documents'):
        print("\nFørste dokument:")
        doc = result['documents'][0]
        print(f"Metadata: {doc.metadata}")
        print(f"Innhold (utdrag): {doc.page_content[:200]}...")
    
    return result

async def main():
    """Kjør alle tester i rekkefølge for å identifisere rotårsaken til problemet"""
    print("STARTER TESTING AV HOVEDGRAFEN\n")
    
    # Test 1: Routeren
    router_result = await test_router()
    
    # Test 2: Forskningsplan
    plan_result = await test_research_plan()
    
    # Test 3: Query-generering
    queries = await test_query_generation()
    
    # Test 4: Direkte retrieval
    docs = await test_retrieval()
    
    # Test 5: Full forskningsprosess
    research_result = await test_full_research()
    
    print("\nTESTING FULLFØRT")

if __name__ == "__main__":
    asyncio.run(main()) 