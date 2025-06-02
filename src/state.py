"""State management for Neo RAG Research Agent.

Forenklet state-struktur fokusert på meldinger og dokumenter 
uten kompleks routing-logikk.
"""

from dataclasses import dataclass, field
from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def reduce_docs(left: List[Document], right: List[Document]) -> List[Document]:
    """Kombiner to lister med dokumenter og fjerning av duplikater.
    
    Args:
        left: Eksisterende dokumentliste
        right: Nye dokumenter som skal legges til
        
    Returns:
        Kombinert liste uten duplikater
    """
    if not right:
        return left
    if not left:
        return right
    
    # Bruk page_content som nøkkel for å unngå duplikater
    seen_content = {doc.page_content for doc in left}
    new_docs = [doc for doc in right if doc.page_content not in seen_content]
    
    return left + new_docs


@dataclass(kw_only=True)
class InputState:
    """Input state for agenten.

    Definerer strukturen for input state, som inkluderer
    meldingene utvekslet mellom bruker og agent. Fungerer som
    en begrenset versjon av full State for å gi et smalere grensesnitt
    til omverdenen sammenlignet med det som vedlikeholdes internt.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    """Meldinger sporer hovedutføringsstate for agenten.

    Typisk samler dette et mønster av Human/AI/Human/AI meldinger; hvis
    du skulle kombinere denne templaten med et tool-calling ReAct agent pattern,
    kan det se slik ut:

    1. HumanMessage - bruker input
    2. AIMessage med .tool_calls - agent velger tool(s) å bruke for å samle
         informasjon
    3. ToolMessage(s) - responsene (eller feilene) fra utførte tools
    
        (... gjenta steg 2 og 3 etter behov ...)
    4. AIMessage uten .tool_calls - agent responderer i ustrukturert
        format til brukeren.

    5. HumanMessage - bruker responderer med neste samtaletura.

        (... gjenta steg 2-5 etter behov ... )
    
    Kombinerer to lister med meldinger, oppdaterer eksisterende meldinger med ID.

    Som standard sikrer dette at state er "append-only", med mindre den
    nye meldingen har samme ID som en eksisterende melding.

    Returns:
        En ny liste med meldinger med meldingene fra `right` sammensmeltet inn i `left`.
        Hvis en melding i `right` har samme ID som en melding i `left`, vil
        meldingen fra `right` erstatte meldingen fra `left`."""


@dataclass(kw_only=True)
class AgentState(InputState):
    """Forenklet state for Neo RAG Research Agent.
    
    Fokusert på essensielle elementer: meldinger og dokumenter.
    Fjernet kompleks routing-logikk for elegant tool-basert tilnærming.
    """

    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populert av tools. Dette er en liste med dokumenter som agenten kan referere til."""

    # Fjernet for forenkling:
    # - router: Router (kompleks 3-veis klassifisering)  
    # - steps: list[str] (forskningsplan-steg)
    # 
    # Den nye tilnærmingen bruker naturlig tool-valg i stedet for forhåndsdefinert routing. 