"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

from typing import Any, Dict, List, Literal, TypedDict, cast
from langgraph.graph import END, START, StateGraph

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langsmith.run_helpers import traceable

from retrieval_graph import prompts
from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router, reduce_docs
from shared.utils import format_docs, load_chat_model


@traceable(run_type="chain")
async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    
    # Log klassifiseringsprosessen
    print(f"Router initialisert med spørsmål: {state.messages[-1].content}")
    
    try:
        response = cast(
            Router, await model.with_structured_output(Router).ainvoke(messages)
        )
        # Sikre at response har både 'type' og 'logic'
        if 'type' not in response:
            print("Advarsel: Klassifisering mangler 'type', bruker standardverdi 'lovspørsmål'")
            response['type'] = "lovspørsmål"  # Bruk lovspørsmål som standard
        if 'logic' not in response:
            print("Advarsel: Klassifisering mangler 'logic', bruker tom streng")
            response['logic'] = ""
            
        # Spesiell sjekk for lovrelaterte spørsmål som blir feilklassifisert
        last_message = state.messages[-1].content.lower() if state.messages else ""
        if (response['type'] == "generelt" and 
            any(word in last_message for word in ["lov", "lovverk", "paragraf", "forskrift", "juss", "rett", "rettslig", "innsyn", "offentlig"])):
            print(f"Reklassifiserer fra 'generelt' til 'lovspørsmål' basert på nøkkelord i spørsmålet: {last_message}")
            response['type'] = "lovspørsmål"
            response['logic'] = f"Spørsmålet inneholder lovrelaterte nøkkelord: {last_message}"
            
    except Exception as e:
        # Hvis noe går galt med strukturert output, bruk standardverdier
        print(f"Error in analyze_and_route_query: {e}")
        response = Router(type="lovspørsmål", logic="Feilsituasjon oppstod, bruker lovspørsmål som standard")
    
    print(f"Router klassifisering: type={response['type']}, logic={response['logic']}")
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "lovspørsmål":
        return "create_research_plan"
    elif _type == "mer-info":
        return "ask_for_more_info"
    elif _type == "generelt":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


@traceable(run_type="chain")
async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


@traceable(run_type="chain")
async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


@traceable(run_type="chain")
async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    response = cast(Plan, await model.ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}


@traceable(run_type="chain")
async def conduct_research(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.
        config (RunnableConfig): Configuration for the retriever and models.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    # Forskergraf er ikke en kjørbar graf, men en StateGraph definisjon
    # Vi må bruke vår egen forsknings-logikk her
    
    from retrieval_graph.researcher_graph.state import ResearcherState
    from shared.retrieval import make_retriever
    import logging
    
    # Sett opp logging for denne funksjonen
    logger = logging.getLogger("conduct_research")
    logger.setLevel(logging.DEBUG)  # Sett loggingsnivå til DEBUG for mer detaljer
    logger.info(f"Starter forskningsprosess på steg: {state.steps[0]}")
    
    # Sjekk miljøvariabler
    import os
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
    
    logger.info(f"Miljøvariabler: PINECONE_API_KEY {'funnet' if pinecone_api_key else 'MANGLER'}")
    logger.info(f"Miljøvariabler: OPENAI_API_KEY {'funnet' if openai_api_key else 'MANGLER'}")
    logger.info(f"Bruker Pinecone indeks: {pinecone_index_name}")
    
    # Initialize researcher state with the first step as question
    researcher_state = ResearcherState(question=state.steps[0])
    
    # Generate queries based on the question
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    
    class Response(TypedDict):
        queries: list[str]

    structured_model = model.with_structured_output(Response)
    messages = [
        {"role": "system", "content": configuration.generate_queries_system_prompt},
        {"role": "human", "content": researcher_state.question},
    ]
    response = cast(Response, await structured_model.ainvoke(messages))
    queries = response["queries"]
    logger.info(f"Genererte {len(queries)} spørringer: {queries}")
    
    # Run retrieval for each query
    all_docs = []
    try:
        with make_retriever(config) as retriever:
            # Logg retriever-informasjon
            logger.info(f"Retriever type: {type(retriever)}")
            if hasattr(retriever, 'vectorstore'):
                logger.info(f"VectorStore type: {type(retriever.vectorstore)}")
                
                # Sjekk content_key
                if hasattr(retriever.vectorstore, '_content_key'):
                    logger.info(f"VectorStore content_key: {retriever.vectorstore._content_key}")
                else:
                    logger.warning("VectorStore mangler _content_key attributt")
                    
                # Sjekk text_key
                if hasattr(retriever.vectorstore, '_text_key'):
                    logger.info(f"VectorStore text_key: {retriever.vectorstore._text_key}")
                else:
                    logger.warning("VectorStore mangler _text_key attributt")
            
            # Sikre at content_key er satt til "text"
            if hasattr(retriever, 'vectorstore') and hasattr(retriever.vectorstore, '_content_key'):
                if retriever.vectorstore._content_key != "text":
                    logger.warning(f"Retriever bruker content_key={retriever.vectorstore._content_key}, men forventer 'text'. Setter det til 'text'...")
                    retriever.vectorstore._content_key = "text"
                    logger.info(f"Satt content_key til 'text', ny verdi: {retriever.vectorstore._content_key}")
            
            for query in queries:
                logger.info(f"Utfører søk med spørring: {query}")
                docs = await retriever.ainvoke(query, config)
                logger.info(f"Søk resulterte i {len(docs)} dokumenter")
                
                # Logg mer detaljert om de første par dokumentene
                for i, doc in enumerate(docs[:2]):
                    logger.info(f"Dokument {i} type: {type(doc)}")
                    logger.info(f"Dokument {i} attributter: {dir(doc)}")
                    
                    # Sjekk alle nøkkelattributter
                    if hasattr(doc, 'page_content'):
                        logger.info(f"Dokument {i} har page_content ({len(doc.page_content)} tegn)")
                        logger.debug(f"page_content innhold: {doc.page_content[:100]}...")
                    else:
                        logger.warning(f"Dokument {i} mangler page_content")
                        
                    if hasattr(doc, 'text'):
                        logger.info(f"Dokument {i} har text-felt ({len(doc.text)} tegn)")
                        logger.debug(f"text innhold: {doc.text[:100]}...")
                    else:
                        logger.warning(f"Dokument {i} mangler text-felt")
                    
                    # Sjekk metadata
                    metadata_summary = {}
                    if hasattr(doc, 'metadata'):
                        metadata_summary = {k: v for k, v in doc.metadata.items() if k in ['lov_id', 'lov_navn', 'paragraf_nr', 'kapittel_nr']}
                        logger.info(f"Dokument {i} metadata: {metadata_summary}")
                    else:
                        logger.warning(f"Dokument {i} mangler metadata")
                
                all_docs.extend(docs)
    except Exception as e:
        logger.error(f"Feil under retrieval: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Her bruker vi reduce_docs for å fjerne duplikater, 
    # men må kalle den riktig, først med None og deretter med dokumentene
    logger.info(f"Totalt {len(all_docs)} dokumenter før deduplisering")
    unique_docs = reduce_docs(None, all_docs)
    logger.info(f"Totalt {len(unique_docs)} dokumenter etter deduplisering")
    
    # Return updated state
    remaining_steps = state.steps[1:] if len(state.steps) > 1 else []
    return {"documents": unique_docs, "steps": remaining_steps}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


@traceable(run_type="chain")
async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node(respond)
builder.add_node(create_research_plan)

# Add edges
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges(
    "analyze_and_route_query",
    route_query,
    {
        "ask_for_more_info": "ask_for_more_info",
        "respond_to_general_query": "respond_to_general_query",
        "create_research_plan": "create_research_plan",
    },
)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges(
    "conduct_research", check_finished, {"respond": "respond", "conduct_research": "conduct_research"}
)
builder.add_edge("respond", END)
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)

# Eksporter grafen for bruk av andre moduler
graph = builder.compile()

# Wrap grafen med LangSmith tracing
original_ainvoke = graph.ainvoke

# Oppdatere traced_ainvoke for å bruke traceable
@traceable(run_type="chain", name="full_retrieval_graph")
async def traced_ainvoke(state, config=None):
    """Wrap the graph's invoke method with LangSmith tracing."""
    if config is None:
        config = {}
    return await original_ainvoke(state, config)

# Erstatt opprinnelig ainvoke med traced versjon
graph.ainvoke = traced_ainvoke
