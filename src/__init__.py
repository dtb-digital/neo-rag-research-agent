"""Neo RAG Research Agent - Hovedmodul.

Forenklet juridisk assistent for norsk lovgivning med elegant tool-basert arkitektur.

Ny arkitektur:
- 2 noder: lovdata_agent + tools  
- 4 selvstendige tools: sok_lovdata, generer_sokestrenger, hent_lovtekst, sammenstill_svar
- Naturlig tool-valg uten kompleks routing
- Ren state: kun meldinger og dokumenter

Eksport:
- graph: Hovedgraf for juridiske spørsmål
- AgentState, InputState: State-klasser
- traced_ainvoke: Traced wrapper for grafen
"""

from src.neo_rag_agent import graph, traced_ainvoke, AgentState, InputState

__all__ = ["graph", "traced_ainvoke", "AgentState", "InputState"] 