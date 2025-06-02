"""Neo RAG Research Agent - Juridisk assistent for norsk lovgivning.

Dette er hovedmodulet for Neo RAG Research Agent med elegant 2-node arkitektur:
1. lovdata_agent - Hovedassistent med intelligent tool-valg  
2. tool_node - Utfører de fire spesialiserte tools

Erstatter kompleks routing-system med intelligent tool-valg.
"""

from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith.run_helpers import traceable

from src.config import AgentConfiguration
from src.state import AgentState, InputState
from src.utils import load_chat_model
from src.tools import TOOLS


@traceable(run_type="chain")
async def lovdata_agent(state: AgentState, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """Hovedassistent for juridisk informasjon med naturlig tool-valg.
    
    Denne agenten velger intelligent mellom fire tilgjengelige tools:
    - sok_lovdata: Grunnleggende vektorsøk i Pinecone
    - generer_sokestrenger: Generer flere søkestrenger for komplekse spørsmål
    - hent_lovtekst: Direkte henting av spesifikke lovtekster
    - sammenstill_svar: Sammenstill endelig svar fra innsamlede dokumenter
    
    Args:
        state: Nåværende agent state med meldinger og dokumenter
        config: Konfigurasjon med modell-innstillinger
        
    Returns:
        dict med 'messages' som inneholder agent-respons (med tool calls eller endelig svar)
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).bind_tools(TOOLS)
    
    # System prompt for intelligent tool-valg
    system_prompt = """Du er en juridisk assistent som hjelper med norsk lovgivning.

Du har tilgang til disse fire tools:

1. **sok_lovdata(query, k)** - Søk i Lovdata med vektorsøk
   - Bruk for grunnleggende søk etter juridisk informasjon
   - Kan kalles flere ganger med forskjellige søkestrenger

2. **generer_sokestrenger(question, num_queries)** - Generer målrettede søkestrenger
   - Bruk for komplekse/brede spørsmål som trenger flere perspektiver
   - Kall deretter sok_lovdata() for hver genererte query

3. **hent_lovtekst(lov_id, paragraf_nr, kapittel_nr)** - Hent spesifikke lovtekster
   - Bruk når du har identifisert spesifikke lover fra metadata i søkeresultater
   - Bruk for presise juridiske referanser

4. **sammenstill_svar(documents, original_question)** - Sammenstill endelig svar
   - Bruk når du har samlet alle relevante dokumenter
   - Gir strukturert juridisk svar med kildehenvisninger

**Strategier:**

For **enkle spørsmål**:
1. sok_lovdata() → sammenstill_svar()

For **komplekse spørsmål**: 
1. generer_sokestrenger() → sok_lovdata() (flere ganger) → sammenstill_svar()

For **spesifikke lovoppslag**:
1. sok_lovdata() (for å få lov_id) → hent_lovtekst() → sammenstill_svar()

**Viktig**: Kall alltid sammenstill_svar() som siste steg for å gi brukeren et strukturert svar."""

    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


def should_call_tool(state: AgentState) -> Literal["tools", "__end__"]:
    """Bestem om agenten skal kalle tools eller avslutte samtalen.
    
    Args:
        state: Nåværende agent state
        
    Returns:
        "tools" hvis agent har tool calls, "__end__" hvis endelig svar er gitt
    """
    last_message = state.messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "__end__"


def create_graph() -> StateGraph:
    """Lag den forenklede 2-node grafen.
    
    Returns:
        Kompilert StateGraph med agent og tool nodes
    """
    # Initialiser workflow
    workflow = StateGraph(AgentState, input=InputState)
    
    # Legg til nodes
    workflow.add_node("lovdata_agent", lovdata_agent)
    workflow.add_node("tools", ToolNode(TOOLS))
    
    # Legg til edges
    workflow.add_edge(START, "lovdata_agent")
    workflow.add_conditional_edges(
        "lovdata_agent",
        should_call_tool,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    workflow.add_edge("tools", "lovdata_agent")
    
    return workflow


# Lag og kompiler grafen
graph = create_graph().compile()


@traceable(run_type="chain", name="neo_rag_agent")
async def traced_ainvoke(state, config=None):
    """Traced wrapper for Neo RAG Agent.
    
    Args:
        state: Input state
        config: Valgfri konfigurasjon
        
    Returns:
        Endelig state etter graf-utførelse
    """
    return await graph.ainvoke(state, config)


# Eksporter traced graph som hovedgrensesnitt
__all__ = ["traced_ainvoke", "graph", "AgentState", "InputState"] 