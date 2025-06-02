---
description: Refaktorering av Neo RAG Research Agent fra kompleks routing til elegant tool-basert arkitektur
status: I utvikling
---

# Neo RAG Research Agent - Refaktoring PRD

## Målsetting
Forenkle agenten fra kompleks routing-basert system (6+ noder) til elegant tool-basert arkitektur (2 noder: agent + tool_node) hvor intelligensen ligger i naturlig tool-valg i stedet for forhåndsdefinert routing-logikk.

## Nåværende problemer
- **Overflødig routing-logikk**: 3-veis klassifisering (lovspørsmål/mer-info/generelt) med kunstig beslutningsprosess
- **Unødvendig kompleks graf-struktur**: 6+ noder med undergraf (researcher_graph)
- **Redundante retriever-providers**: Kun Pinecone brukes som vektorbase
- **Kunstig research plan konsept**: Steps-håndtering som kunne vært naturlig tool-valg

## Ny arkitektur

### Graf-struktur (Target)
```python
workflow = StateGraph(State)
workflow.add_node("lovdata_agent", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_edge(START, "lovdata_agent")
workflow.add_conditional_edges(
    "lovdata_agent",
    should_call_tool,
    {
        "tool_node": "tool_node",
        "END": END
    }
)
workflow.add_edge("tool_node", "lovdata_agent")
```

### Agent-tilstand
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: Annotated[list[Document], reduce_docs]
    # Fjernet: router, steps, andre routing-relaterte felter
```

## Tool-definitioner (Endelig)

### 1. `sok_lovdata(query: str, k: int = 5) -> List[Document]`
- **Formål**: Grunnleggende vektorsøk i Pinecone-indeksen
- **Når brukes**: 
  - For enkle, direkte søk etter juridisk informasjon
  - Som byggekloss når agenten kjører multiple søk
  - Første steg i alle søkestrategier
- **Input**: En søkestreng og antall resultater som skal returneres
- **Output**: Liste med relevante dokumenter med metadata
- **Note**: Agenten kan kalle denne flere ganger med ulike queries for å bygge bredere kunnskapsbase

### 2. `generer_sokestrenger(question: str, num_queries: int = 3) -> List[str]`
- **Formål**: Intelligent oppbreking av komplekse juridiske spørsmål til målrettede søkestrenger
- **Når brukes**: 
  - For brede eller flertydige spørsmål som krever flere perspektiver
  - Når agenten vurderer at ett søk ikke dekker alle relevante aspekter
  - For å få mer omfattende juridisk dekning av et tema
- **Input**: Brukerens opprinnelige spørsmål og ønsket antall queries
- **Output**: Liste med genererte, varierte søkestrenger
- **Arbeidsflyt**: Agenten kaller deretter `sok_lovdata()` for hver genererte query

### 3. `hent_lovtekst(lov_id: str, paragraf_nr: Optional[str] = None, kapittel_nr: Optional[str] = None) -> Document`
- **Formål**: Direkte henting av spesifikke lovtekster basert på eksakte identifikatorer
- **Når brukes**:
  - Når agenten har identifisert spesifikke lover fra metadata i søkeresultater
  - For å hente komplette lovtekster når brukeren spør om spesifikke paragrafer
  - Som oppfølging til vektorsøk for å få mer presise juridiske referanser
- **Input**: Lovens ID (påkrevd), paragraf- og kapittelnummer (valgfri for filtrering)
- **Output**: Komplett lovtekst-dokument med full metadata
- **Integrasjon**: Bruker metadata fra `sok_lovdata()` resultater for målrettede oppslag

### 4. `sammenstill_svar(documents: List[Document], original_question: str) -> str`
- **Formål**: Sammenstilling av kvalitetssvar basert på alle innsamlede juridiske dokumenter
- **Når brukes**: 
  - Etter at alle relevante søk og dokumenthenting er gjennomført
  - For å kombinere informasjon fra multiple kilder til et helhetlig svar
  - For å gi strukturerte, juridisk korrekte svar på komplekse spørsmål
- **Input**: Alle relevante dokumenter og brukerens opprinnelige spørsmål
- **Output**: Ferdig formulert juridisk svar med kildehenvisninger og metadata
- **Ekstra funksjonalitet**: Identifiserer og kommuniserer hvis informasjonen er utilstrekkelig for å svare

## Fordeler med denne tilnærmingen
- **Fleksibilitet**: Agenten kan dynamisk velge mellom enkelt søk, kompleks research, eller direktehenting
- **Modularitet**: Hver tool har tydelig ansvarsområde uten overlapp eller redundans
- **Testbarhet**: Hver tool kan testes og utvikles isolert
- **Evolverbarhet**: Enkelt å legge til nye juridiske søkestrategier eller datakilder
- **Gjenbrukbarhet**: `sok_lovdata()` og `hent_lovtekst()` brukes som byggeklosser i alle strategier

## Eksempel på agent-arbeidsflyt

### Enkelt spørsmål: "Hva er fraværsgrensen i videregående?"
1. Agent vurderer spørsmålet som enkelt og spesifikt
2. Kaller `sok_lovdata("fraværsgrense videregående")`
3. Kaller `sammenstill_svar(documents, original_question)`
4. Returnerer strukturert svar med kildehenvisninger

### Komplekst spørsmål: "Hvordan påvirker GDPR norske bedrifters håndtering av kundedata?"
1. Agent vurderer spørsmålet som bredt og komplekst
2. Kaller `generer_sokestrenger()` → får ["GDPR norsk lovgivning", "bedrifter personvernkrav", "datatilsyn sanksjoner"]
3. Kaller `sok_lovdata()` tre ganger med hver query
4. Kaller `sammenstill_svar()` med alle innsamlede dokumenter
5. Returnerer omfattende, strukturert svar med multiple kilder

### Spesifikt lovoppslag: "Kan du hente § 15 fra personvernloven?"
1. Agent identifiserer dette som forespørsel om spesifikk lovtekst
2. Kaller først `sok_lovdata("personvernloven § 15")` for å få lov_id
3. Bruker metadata til å kalle `hent_lovtekst(lov_id="personvern-lov-2018", paragraf_nr="15")`
4. Kaller `sammenstill_svar()` med den eksakte lovteksten
5. Returnerer presis paragraf med full juridisk kontekst

### Iterativ strategi for usikre svar:
1. Enkelt søk med `sok_lovdata()`
2. `sammenstill_svar()` → identifiserer hull i informasjonen eller behov for spesifikke lovtekster
3. `generer_sokestrenger()` eller `hent_lovtekst()` for å målrette manglende informasjon
4. Nye søk med `sok_lovdata()` eller direkte lovteksthenting
5. Endelig `sammenstill_svar()` med komplett juridisk grunnlag

## Beholder fra eksisterende system
- **Pinecone-integrasjon og søkelogikk**: `make_retriever(config)`, query-generering, deduplisering
- **LangSmith tracing infrastruktur**: `@traceable` decorator og tracing-wrapper
- **Avansert søkefunksjonalitet**: Fra `conduct_research()` - parallelle søk, feilhåndtering, dokumenthåndtering
- **Respons-generering**: Fra `respond()` - dokumentformatering til prompts, metadata-håndtering
- **Konfigurasjonssystem**: `AgentConfiguration.from_runnable_config()`, `load_chat_model()`
- **Omfattende logging**: Debug-infrastruktur og feilrapportering
- **Prompt-templating**: Eksisterende system for prompt-formatering
- **Metadata-struktur**: Eksisterende lovdata-spesifikke metadata-felter

## Fjerner fra eksisterende system
- **Hele router-systemet**: `analyze_and_route_query()`, `route_query()`, Router-klassen
- **Research plan konsept**: `create_research_plan()`, `check_finished()`, steps-håndtering
- **Separate svar-noder**: `ask_for_more_info()`, `respond_to_general_query()`
- **Kompleks graf-struktur**: Fra 6 noder + conditional edges til 2 noder
- **Researcher_graph integrasjon**: Som egen subgraph
- **Index_graph modulen**: Allerede fjernet
- **Unødvendige tilstandsvariabler**: (`steps`, routing-relaterte felter)

## Gjenbruk i ny tool-arkitektur

### MCP-server som kode-referanse (IKKE som dependency)
```python
# KLARGJØRING: MCP-server koden brukes som TEMPLATE/INSPIRASJON
# Vi kopierer og tilpasser relevant logikk direkte inn i hver tool
# Hver tool blir selvstendige, komplette funksjoner uten kryss-referanser

# Fra MCP-server - KOPIERER Pinecone-konfigurasjon og søkelogikk:
@tool
async def sok_lovdata(query: str, k: int = 5) -> List[Document]:
    """Søk i Lovdata med vektorsøk. Komplett, selvstedig implementasjon."""
    
    # All nødvendig logikk direkte i tool-en (inspirert av MCP-server)
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index("lovdata-embedding-index")
    
    # Embed query (samme pattern som MCP-server)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = await embeddings.aembed_query(query)
    
    # Search Pinecone (kopierer søkelogikk)
    search_results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True
    )
    
    # Format til Document objekter (samme metadata-struktur)
    documents = []
    for match in search_results.matches:
        doc = Document(
            page_content=match.metadata.get("content", ""),
            metadata={
                "lov_id": match.metadata.get("lov_id"),
                "paragraf_nr": match.metadata.get("paragraf_nr"),
                "lov_navn": match.metadata.get("lov_tittel"),
                "score": match.score
            }
        )
        documents.append(doc)
    
    return documents

# Fra MCP-server - KOPIERER metadata-filter logikk:
@tool
async def hent_lovtekst(lov_id: str, paragraf_nr: Optional[str] = None) -> List[Document]:
    """Hent spesifikke lovtekster med metadata-filtering."""
    
    # Samme Pinecone-oppsett som over
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index("lovdata-embedding-index")
    
    # Bygger filter (kopierer fra MCP-server hent_lovtekst())
    filter_dict = {"lov_id": {"$eq": lov_id}}
    if paragraf_nr:
        filter_dict["paragraf_nr"] = {"$eq": paragraf_nr}
    
    # Søk med metadata-filter
    search_results = index.query(
        vector=[0] * 1536,  # Dummy vector for metadata-only søk
        top_k=50,           # Høyere k for komplette lovtekster
        filter=filter_dict,
        include_metadata=True
    )
    
    # Samme dokumentformatering som over
    documents = []
    for match in search_results.matches:
        doc = Document(
            page_content=match.metadata.get("content", ""),
            metadata={
                "lov_id": match.metadata.get("lov_id"),
                "paragraf_nr": match.metadata.get("paragraf_nr"),
                "kapittel_nr": match.metadata.get("kapittel_nr"),
                "lov_navn": match.metadata.get("lov_tittel")
            }
        )
        documents.append(doc)
    
    return documents
```

### Nye implementasjoner (med OpenAI direkte)
```python
@tool
async def generer_sokestrenger(question: str, num_queries: int = 3) -> List[str]:
    """Generer flere søkestrenger fra ett spørsmål. Komplett implementasjon."""
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "Du genererer varierte søkestrenger for juridisk informasjon i norsk lovdata."
            },
            {
                "role": "user", 
                "content": f"Lag {num_queries} forskjellige søkestrenger for: {question}"
            }
        ]
    )
    
    # Parse respons til liste med strenger
    content = response.choices[0].message.content
    queries = [q.strip('- ').strip() for q in content.split('\n') if q.strip()]
    return queries[:num_queries]  # Sørg for riktig antall

@tool
async def sammenstill_svar(documents: List[Document], original_question: str) -> str:
    """Sammenstill juridisk svar basert på dokumenter. Komplett implementasjon."""
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Format dokumenter for prompt (inspirert av MCP respond())
    docs_text = ""
    for i, doc in enumerate(documents):
        metadata_str = ", ".join([
            f"{k}: {v}" for k, v in doc.metadata.items() 
            if k in ["lov_id", "lov_navn", "paragraf_nr", "kapittel_nr"] and v
        ])
        docs_text += f"\n\nDokument {i+1}:\n{doc.page_content}\nMetadata: {metadata_str}"
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Du er en juridisk assistent som gir presise svar basert på norsk lovgivning."
            },
            {
                "role": "user",
                "content": f"Spørsmål: {original_question}\n\nRelevante juridiske dokumenter:{docs_text}\n\nGi et strukturert svar med kildehenvisninger."
            }
        ]
    )
    
    return response.choices[0].message.content
```

## Ny arkitektur-tilnærming: Selvstendige tools

### Ren arkitektur uten kryss-referanser
```python
# Enkel graf-struktur med fullstendige tools
workflow = StateGraph(AgentState) 
workflow.add_node("lovdata_agent", agent_node)
workflow.add_node("tool_node", ToolNode(tools=[
    sok_lovdata,           # Komplett Pinecone-implementasjon
    hent_lovtekst,         # Komplett metadata-filter implementasjon  
    generer_sokestrenger,  # Komplett query-generering
    sammenstill_svar       # Komplett svar-sammenstilling
]))

# Hver tool er selvstendige uten avhengigheter:
# ✅ All logikk direkte i tool-funksjonen
# ✅ Lett å forstå ved å lese kun den funksjonen  
# ✅ Kan testes isolert uten dependencies
# ✅ Ingen referanser til andre deler av systemet
```

## Implementasjonsplan (Revidert for ren kode)
1. ✅ Definere tool-arkitektur (fire komplementære tools)
2. **Implementere tools som selvstendige funksjoner**:
   - `sok_lovdata()` - **Kopierer Pinecone-logikk** fra MCP-server som inspirasjon
   - `hent_lovtekst()` - **Kopierer metadata-filter logikk** direkte inn i tool
   - `generer_sokestrenger()` - **Ny, komplett implementasjon** med OpenAI API
   - `sammenstill_svar()` - **Ny, komplett implementasjon** med OpenAI API
3. **Lage enkel graf-struktur**: Agent + ToolNode med selvstendige tools
4. **Minimal AgentState**: Beholde messages/documents, fjerne router/steps
5. **Fjerne gammel kode**: Router-system, research plan, alle gamle noder
6. **Validere**: Teste hver tool isolert og i sammenheng

## Arkitektur-fordeler med selvstendige tools
- **Enkel å forstå**: All logikk synlig i hver tool-funksjon
- **Lett å debugge**: Ingen skjulte avhengigheter eller kryss-referanser
- **Lett å teste**: Hver tool kan testes isolert med mock-data
- **Lett å modifisere**: Endringer i en tool påvirker ikke andre
- **Ren kodebase**: Lineær, forutsigbar kode uten komplekse referanser
- **Rask utvikling**: Bruke MCP-server som template gir god start

## Suksesskriterier (Oppdatert for ren kode)
- **Enklere kodebase** med selvstendige, forståelige tools
- **Samme eller bedre respons-kvalitet** ved å bruke beprøvde implementasjonsmønstre
- **Mer fleksibel agent-atferd** med naturlig tool-valg
- **Lettere å vedlikeholde** - hver tool er isolert og komplett
- **Lettere å teste** - ingen komplekse avhengigheter
- **Eliminert kryss-referanser** - all logikk er synlig og direkte
- **Rask implementering** ved å bruke MCP-server som kode-referanse

## MCP-server som kode-referanse
### Hva vi bruker som inspirasjon:
- **Pinecone-konfigurasjon** - API-nøkler, index-navn, embedding-modell
- **Søke-patterns** - Hvordan kalle Pinecone API og OpenAI embeddings
- **Metadata-struktur** - Hvilke felter som finnes (`lov_id`, `paragraf_nr`, etc.)
- **Filter-logikk** - Hvordan bygge metadata-filtre for Pinecone
- **Feilhåndtering** - Hvilke exceptions å fange og hvordan
- **Respons-formatering** - Hvordan strukturere Document-objekter

### Hva vi IKKE gjør:
- ❌ Lage avhengigheter til MCP-server kode
- ❌ Importere funksjoner fra andre moduler  
- ❌ Wrappere rundt eksisterende infrastruktur
- ❌ Komplekse kryss-referanser mellom komponenter

### Implementasjonsplan
1. ✅ Definere tool-arkitektur (fire granulære tools)
2. **Skrive selvstendige tools med MCP-server som kode-template**
3. Lage enkel graf-struktur med standard LangGraph agent + ToolNode
4. Minimal oppdatering av prompts for tool-kontekst  
5. Fjerne all gammel routing-logikk og komplekse noder
6. Teste hver tool isolert og validere full funksjonalitet 