"""
MCP-server for Lovdata RAG-agent.

Implementerer en Model Context Protocol (MCP) server som bruker stdio-transport
for å kommunisere med Claude og eksponere verktøy for å søke i lovdata.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional
import dotenv
from pathlib import Path
import logging

# Sett opp en enkel initiell logger til stderr før vi laster full logging
init_logger = logging.getLogger("init")
init_logger.setLevel(logging.INFO)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
init_logger.addHandler(stderr_handler)

# Last inn miljøvariabler fra .env-filen
init_logger.info("Laster miljøvariabler fra .env-filen...")
dotenv.load_dotenv()

# Legg til prosjektets rotmappe i sys.path for å støtte importer
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
# Legg til src-mappen i sys.path for å støtte begge importstiler
sys.path.append(os.path.join(project_root, "src"))

# Importer logger.py først for å sette opp grunnleggende logging
from shared.logging_config import logger, setup_logger, configure_logging

# Konfigurer logging eksplisitt for å sikre at alt er satt opp korrekt
configure_logging()

# Nå er logging konfigurert, så vi kan fortsette med resten av importen
init_logger.info("Logging konfigurasjon initialisert")

# Hvis kritiske miljøvariabler mangler, prøv å lese direkte fra .env-filen
env_file = Path('.env')
if env_file.exists() and (
    not os.environ.get("PINECONE_API_KEY") or 
    not os.environ.get("OPENAI_API_KEY") or
    not os.environ.get("LANGSMITH_API_KEY")
):
    init_logger.info("Noen miljøvariabler mangler, prøver å lese direkte fra .env-filen...")
    try:
        env_content = env_file.read_text()
        
        # Hent nøkler med regex
        import re
        pinecone_match = re.search(r'PINECONE_API_KEY=(.+)', env_content)
        openai_match = re.search(r'OPENAI_API_KEY=(.+)', env_content)
        langsmith_api_match = re.search(r'LANGSMITH_API_KEY=(.+)', env_content)
        langsmith_tracing_match = re.search(r'LANGSMITH_TRACING=(.+)', env_content)
        langsmith_project_match = re.search(r'LANGSMITH_PROJECT=(.+)', env_content)
        langsmith_endpoint_match = re.search(r'LANGSMITH_ENDPOINT=(.+)', env_content)
        
        # Sett miljøvariablene direkte
        if pinecone_match and not os.environ.get("PINECONE_API_KEY"):
            os.environ["PINECONE_API_KEY"] = pinecone_match.group(1).strip()
            init_logger.info("Satt PINECONE_API_KEY direkte fra .env")
        
        if openai_match and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_match.group(1).strip()
            init_logger.info("Satt OPENAI_API_KEY direkte fra .env")
            
        # Legg til LANGSMITH-miljøvariablene
        if langsmith_api_match and not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_API_KEY direkte fra .env")
            
        if langsmith_tracing_match and not os.environ.get("LANGSMITH_TRACING"):
            os.environ["LANGSMITH_TRACING"] = langsmith_tracing_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_TRACING direkte fra .env")
            
        if langsmith_project_match and not os.environ.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = langsmith_project_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_PROJECT direkte fra .env")
            
        if langsmith_endpoint_match and not os.environ.get("LANGSMITH_ENDPOINT"):
            os.environ["LANGSMITH_ENDPOINT"] = langsmith_endpoint_match.group(1).strip()
            init_logger.info("Satt LANGSMITH_ENDPOINT direkte fra .env")
    except Exception as e:
        init_logger.error(f"Kunne ikke lese direkte fra .env: {str(e)}")

# Sett standard verdi for Pinecone-indeksen
if not os.environ.get("PINECONE_INDEX_NAME"):
    os.environ["PINECONE_INDEX_NAME"] = "lovdata-paragraf-test"
    init_logger.info("Satt standard PINECONE_INDEX_NAME til 'lovdata-paragraf-test'")

# Sett standard verdi for LANGSMITH_TRACING hvis den ikke er satt
if not os.environ.get("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"
    init_logger.info("Satt standard LANGSMITH_TRACING til 'true'")

# Sett miljøvariabelen LOG_LEVEL til INFO direkte før import av FastMCP
os.environ["LOG_LEVEL"] = "INFO"

# Fortell FastMCP å logge til stderr i stedet for stdout
os.environ["FASTMCP_LOG_TO_STDERR"] = "true"

from fastmcp import FastMCP
from src.utils import truncate_text

# Import retrieval_graph og andre nødvendige moduler
try:
    # Forsøk å importere retrieval_graph for søkefunksjonalitet
    from retrieval_graph import graph as retrieval_graph
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import HumanMessage
except ImportError as e:
    # Stopp programmet hvis retrieval_graph ikke er tilgjengelig
    logger.error(f"Kunne ikke importere retrieval_graph: {str(e)}")
    logger.error("Kan ikke starte MCP-serveren uten retrieval_graph. Avslutter.")
    sys.exit(1)

# Opprett en spesifikk logger for MCP-serveren
mcp_logger = setup_logger("mcp-server")

class LovdataMCPServer:
    """MCP-server for Lovdata RAG-agent."""
    
    def __init__(self):
        """Initialiser MCP-serveren."""
        self.mcp = FastMCP("Lovdata RAG")
        
        # Logg miljøvariabel-status for debugging
        pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "lovdata-paragraf-test")
        langsmith_api_key = os.environ.get("LANGSMITH_API_KEY", "")
        langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "")
        langsmith_project = os.environ.get("LANGSMITH_PROJECT", "")
        
        mcp_logger.info(f"Miljøvariabler: PINECONE_API_KEY {'funnet' if pinecone_api_key else 'MANGLER'}")
        mcp_logger.info(f"Miljøvariabler: OPENAI_API_KEY {'funnet' if openai_api_key else 'MANGLER'}")
        mcp_logger.info(f"Bruker Pinecone indeks: {pinecone_index_name}")
        mcp_logger.info(f"Miljøvariabler: LANGSMITH_API_KEY {'funnet' if langsmith_api_key else 'MANGLER'}")
        mcp_logger.info(f"Miljøvariabler: LANGSMITH_TRACING {'funnet' if langsmith_tracing else 'MANGLER'}")
        mcp_logger.info(f"Miljøvariabler: LANGSMITH_PROJECT {'funnet' if langsmith_project else 'MANGLER'}")
        
        # Sjekk at kritiske miljøvariabler er satt
        if not pinecone_api_key:
            mcp_logger.error("PINECONE_API_KEY er ikke satt! Vektorsøk vil ikke fungere")
        if not openai_api_key:
            mcp_logger.error("OPENAI_API_KEY er ikke satt! Embedding og LLM vil ikke fungere")
        if not langsmith_api_key and langsmith_tracing.lower() == "true":
            mcp_logger.error("LANGSMITH_API_KEY er ikke satt, men LANGSMITH_TRACING er aktivert! LangSmith-sporing vil ikke fungere")
        
        # Registrer verktøy
        self._register_tools()
        
        mcp_logger.info("MCP-server initialisert")
    
    def _register_tools(self):
        """Registrer verktøy for MCP-serveren."""
        
        @self.mcp.tool()
        async def sok_i_lovdata(sporsmal: str, antall_resultater: int = 10) -> str:
            """
            Søk etter relevante lovtekster, forskrifter og juridiske dokumenter basert på ditt spørsmål.
            
            BRUK DETTE VERKTØYET når du ønsker informasjon om juridiske temaer men IKKE kjenner til en spesifikk lov eller paragraf. 
            Dette verktøyet passer best for:
            - Generelle juridiske spørsmål uten referanse til spesifikke lover
            - Når du vil ha en oppsummering eller forklaring av juridiske konsepter
            - Når du ikke vet hvilken lov som regulerer et bestemt tema
            - Når du vil finne relevante lovtekster basert på et tema eller situasjon
            
            Dette verktøyet bruker en avansert juridisk modell til å forstå ditt spørsmål og returnerer 
            de mest relevante delene av norsk lovverk fra Lovdata. Verktøyet analyserer spørsmålet ditt,
            utfører semantisk søk i lovdatabasen, og formaterer resultatene slik at de er lett tilgjengelige.
            
            Args:
                sporsmal: Ditt juridiske spørsmål eller søkeord som du ønsker å finne relevante lover og forskrifter til
                antall_resultater: Antall resultater som skal returneres (standard: 10)
                
            Returns:
                Bearbeidet svar på spørsmålet, eller en liste med relevante lovtekster
                
            Eksempel på bruk:
                "Hva sier offentlighetsloven om innsyn i dokumenter?"
                "Hvilke regler gjelder for permittering av ansatte?"
                "Hva er formålet med offentlighetsloven?"
                "Fortell meg om arbeidsmiljøloven"
                "Jeg trenger informasjon om åndsverkloven"
            """
            mcp_logger.info(f"Utfører søk i lovdata: {sporsmal}, antall_resultater: {antall_resultater}")
            
            try:
                # Konfigurer søk med Pinecone
                config = RunnableConfig(
                    configurable={
                        "retriever_provider": "pinecone",
                        "embedding_model": "openai/text-embedding-3-small",
                        "query_model": "openai/gpt-4o-mini",
                        "response_model": "openai/gpt-4o-mini",
                        "search_kwargs": {"k": antall_resultater}
                    }
                )
                
                # Kjør grafen med spørringen
                mcp_logger.info(f"Invoker retrieval_graph med spørring: {sporsmal}")
                result = await retrieval_graph.ainvoke(
                    {"messages": [HumanMessage(content=sporsmal)]},
                    config,
                )
                
                # Logg resultatet for debugging
                mcp_logger.info(f"Graf-resultat mottatt: {type(result)}")
                
                # Hvis grafen har generert et komplett svar via messages, bruk dette
                if isinstance(result, dict) and 'messages' in result and result['messages']:
                    # Finn siste AI-melding i messages-listen
                    messages = result['messages']
                    mcp_logger.info(f"Mottok {len(messages)} meldinger fra grafen")
                    
                    # Gå gjennom meldingene bakfra for å finne siste AI-melding
                    ai_message = None
                    for msg in reversed(messages):
                        # Sjekk om dette er en AI-melding
                        if (isinstance(msg, dict) and msg.get('type') == 'ai') or (hasattr(msg, 'type') and msg.type == 'ai'):
                            ai_message = msg
                            break
                    
                    if ai_message:
                        # Hent content fra AI-meldingen og returner direkte
                        if hasattr(ai_message, 'content'):
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message.content
                        elif isinstance(ai_message, dict) and 'content' in ai_message:
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message['content']
                
                # Hvis ingen AI-melding ble funnet, returner en feilmelding
                mcp_logger.warning("Ingen AI-melding funnet i resultatet")
                return "Beklager, jeg kunne ikke generere et svar basert på søkeresultatene."
            
            except Exception as e:
                mcp_logger.error(f"Feil ved søk: {str(e)}")
                # Kast feilen videre istedenfor å falle tilbake til dummy-data
                raise e
        
        @self.mcp.tool()
        async def hent_lovtekst(lov_navn: str = "", lov_id: str = "", kapittel_nr: str = "", paragraf_nr: str = "") -> str:
            """
            Hent komplett lovtekst eller forskrift basert på navn, ID, kapittel eller paragraf.
            
            BRUK DETTE VERKTØYET når du vil hente EKSAKT lovtekst og kjenner til en spesifikk lov eller paragraf. 
            Dette verktøyet passer best for:
            - Når du vil sitere eller vise selve lovteksten direkte
            - Når du kjenner navnet på loven (som "Forvaltningsloven" eller "Naturmangfoldloven")
            - Når du vil se en hel lov, et spesifikt kapittel, eller en spesifikk paragraf
            - Når brukeren eksplisitt ber om lovteksten eller paragrafen, ikke bare en forklaring
            
            Dette verktøyet henter en lov, et kapittel eller en paragraf fra lovdata basert på metadata.
            Du kan angi en eller flere parametere for å spesifisere hva du ønsker å hente.
            
            Args:
                lov_navn: Lovens navn (f.eks. "Grunnloven", "Forvaltningsloven", "Offentlighetsloven")
                lov_id: Lovens unike identifikator (f.eks. "lov-1814-05-17-1" for Grunnloven)
                kapittel_nr: Kapittelnummer innen en lov
                paragraf_nr: Paragrafnummer innen en lov
                
            Returns:
                Lovtekst som matcher søkekriteriene
                
            Eksempel på bruk:
                Hent hele Grunnloven: lov_navn="Grunnloven"
                Hent hele Forvaltningsloven: lov_navn="Forvaltningsloven"
                Hent kapittel 3 i Offentlighetsloven: lov_navn="Offentlighetsloven", kapittel_nr="3"
                Hent paragraf 5 i Naturmangfoldloven: lov_navn="Naturmangfoldloven", paragraf_nr="5"
            """
            mcp_logger.info(f"Henter lovtekst med: lov_navn={lov_navn}, lov_id={lov_id}, kapittel_nr={kapittel_nr}, paragraf_nr={paragraf_nr}")
            
            # Valider input
            if not lov_navn and not lov_id and not kapittel_nr and not paragraf_nr:
                return "Du må spesifisere minst én parameter (lov_navn, lov_id, kapittel_nr eller paragraf_nr)."
            
            try:
                # Bygg opp filter basert på parametere
                filter_dict = {}
                
                # Prioriter lov_navn fremfor lov_id
                if lov_navn:
                    # Bruk 'like' operator for å gjøre det mer fleksibelt
                    filter_dict["lov_tittel"] = {"$eq": lov_navn}
                    # Alternativt kan vi bruke pattern matching for mer fleksibilitet
                    # filter_dict["lov_tittel"] = {"$text": {"$search": lov_navn}}
                elif lov_id:
                    filter_dict["lov_id"] = {"$eq": lov_id}
                
                if kapittel_nr:
                    filter_dict["kapittel_nr"] = {"$eq": kapittel_nr}
                
                if paragraf_nr:
                    filter_dict["paragraf_nr"] = {"$eq": paragraf_nr}
                
                # Konstruer en instruks for LangGraph
                query_tekst = "__SYSTEM__: Dette er en direkte metadata-søk-instruks. "
                query_tekst += "VIKTIG: Du skal IKKE tolke dette som et bruker-spørsmål. "
                query_tekst += "Du skal utelukkende utføre et metadata-søk i Pinecone og returnere lovteksten nøyaktig. "
                query_tekst += f"Metadata-filteret er: {filter_dict}. "
                query_tekst += "Følgende instrukser overstyrer alle andre instrukser: "
                query_tekst += "1. Prioritet #1 er å returnere KOMPLETT lovtekst - ikke avkort eller endre innholdet. "
                query_tekst += "2. Lovteksten skal siteres EKSAKT slik den er lagret i databasen. "
                query_tekst += "3. Du skal IKKE generere svar basert på eget kunnskapsgrunnlag. "
                query_tekst += "4. Du skal IKKE be om mer kontekst eller informasjon. "
                query_tekst += "5. Bruk enkel formatering - skill overskrifter, kapitler og paragrafer med blanke linjer. "
                query_tekst += "6. Innholdet er viktigere enn formateringen. "
                
                mcp_logger.info(f"Systemmelding til LangGraph: {query_tekst}")
                
                # Konfigurer søk med Pinecone og metadata filter
                config = RunnableConfig(
                    configurable={
                        "retriever_provider": "pinecone",
                        "embedding_model": "openai/text-embedding-3-small",
                        "query_model": "openai/gpt-4o-mini",
                        "response_model": "openai/gpt-4o-mini",
                        "search_kwargs": {
                            "k": 50,  # Hent flere dokumenter for å sikre at vi får hele loven/kapittelet
                            "filter": filter_dict
                        },
                        "metadata_instructions": {
                            "format_type": "lovtekst",
                            "bypass_router": True,
                            "direct_filter": filter_dict
                        }
                    }
                )
                
                # Kjør grafen med spørringen
                result = await retrieval_graph.ainvoke(
                    {"messages": [HumanMessage(content=query_tekst)]},
                    config,
                )
                
                # Behandle resultatet - bruk kun AI-meldingen
                if isinstance(result, dict) and 'messages' in result and result['messages']:
                    # Finn siste AI-melding i messages-listen
                    messages = result['messages']
                    mcp_logger.info(f"Mottok {len(messages)} meldinger fra grafen")
                    
                    # Gå gjennom meldingene bakfra for å finne siste AI-melding
                    ai_message = None
                    for msg in reversed(messages):
                        if (isinstance(msg, dict) and msg.get('type') == 'ai') or (hasattr(msg, 'type') and msg.type == 'ai'):
                            ai_message = msg
                            break
                    
                    if ai_message:
                        # Hent content fra AI-meldingen og returner direkte
                        if hasattr(ai_message, 'content'):
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message.content
                        elif isinstance(ai_message, dict) and 'content' in ai_message:
                            mcp_logger.info("Returnerer content fra AI-melding")
                            return ai_message['content']
                
                # Hvis ingen AI-melding ble funnet, returner en feilmelding
                mcp_logger.warning("Kunne ikke finne lovtekst via LangGraph-agent")
                return f"Beklager, jeg kunne ikke finne lovtekst med de angitte kriteriene. Sjekk at lov_navn, lov_id, kapittel_nr og paragraf_nr er korrekte."
            
            except Exception as e:
                mcp_logger.error(f"Feil ved henting av lovtekst: {str(e)}")
                # Kast feilen videre
                raise e
        
        @self.mcp.tool()
        async def analyser_juridisk_sporsmal(sporsmal: str, kontekst: str = "") -> str:
            """
            Analyserer et juridisk spørsmål og identifiserer relevante rettsområder og rettskilder.
            
            BRUK DETTE VERKTØYET NÅR:
            - Brukeren ber om juridisk hjelp, analyse, vurdering eller råd
            - Brukeren stiller et spørsmål om lovlighet, rettigheter eller plikter
            - Brukeren spør "hva sier loven om X?" eller lignende
            - Brukeren vil vite om noe er lovlig, ulovlig eller hvordan reglene skal tolkes
            - Brukeren trenger hjelp til å forstå juridiske problemstillinger
            - Enhver forespørsel om juridisk analyse eller vurdering av et rettsspørsmål
            - Brukeren nevner "juridisk vurdering" eller "rettslig vurdering" 
            - Brukeren ber om å få vurdert et juridisk problem eller en rettslig situasjon
            - Bruker nevner "rettsposisjon", "rettsstilling" eller "rettslig standpunkt"
            - Brukeren ber om hjelp til tolkning av lover, forskrifter eller juridiske dokumenter
            - Brukeren ønsker juridisk argumentasjon for eller imot et standpunkt
            
            Dette er ALLTID det første steget i en juridisk analyse og skal brukes før andre 
            juridiske verktøy som juridisk_rettsanvendelse, djevelens_advokat eller juridisk_tankerekke.
            
            Eksempel på triggere:
            - "Kan du hjelpe meg med en juridisk analyse?"
            - "Jeg lurer på om jeg har rett til X"
            - "Er det lovlig å..."
            - "Hva sier loven om..."
            - "Hvilke rettigheter har jeg når..."
            - "Kan jeg bli straffet for å..."
            
            Dette verktøyet hjelper med å:
            - Identifisere konkrete juridiske problemstillinger
            - Finne de relevante rettsområdene (f.eks. privatrett, offentlig rett, arbeidsrett)
            - Avklare hvilke lover og forskrifter som er mest relevante for spørsmålet
            - Strukturere den juridiske analysen
            
            Args:
                sporsmal: Det juridiske spørsmålet som skal analyseres
                kontekst: Ytterligere bakgrunnsinformasjon om situasjonen (valgfritt)
            
            Returns:
                Strukturert analyse med identifiserte problemstillinger og relevante rettskilder
            """
            mcp_logger.info(f"Utfører juridisk problemanalyse: {sporsmal}")
            
            try:
                system_prompt = """
                Du er en erfaren juridisk ekspert med spesialkompetanse i norsk rett og juridisk metode.
                
                Din oppgave er å analysere et juridisk spørsmål og identifisere:
                
                1. PRESISE PROBLEMSTILLINGER: Omformuler spørsmålet til konkrete juridiske problemstillinger
                   - Fokuser på det juridiske kjerneproblemet
                   - Skill mellom hovedproblemstillinger og underproblemstillinger
                   - Bruk juridisk korrekt terminologi
                
                2. RETTSOMRÅDE: Identifiser hvilke rettsområder spørsmålet berører
                   - Privatrett: avtalerett, kontraktsrett, erstatningsrett, familierett, etc.
                   - Offentlig rett: forvaltningsrett, miljørett, helserett, etc.
                   - Strafferett: forbrytelser, strafferammer, straffeprosess
                   - Prosessrett: sivilprosess, tvisteløsning
                   - Spesialområder: arbeidsrett, skatterett, immaterialrett, personvernrett, etc.
                
                3. RELEVANTE RETTSKILDER: Identifiser hvilke primære rettskilder som er aktuelle
                   - Lov og forskrift: Hvilke konkrete lover og paragrafer er relevante?
                   - Forarbeider: Er forarbeidene viktige for tolkning av lovbestemmelsene?
                   - Rettspraksis: Finnes det relevante høyesterettsdommer eller underrettspraksis?
                   - Forvaltningspraksis: Er myndighetspraksis relevant?
                   - Sedvanerett: Finnes det sedvanerett på området?
                   - Juridisk teori: Er det relevant juridisk litteratur?
                   - Reelle hensyn: Hvilke formålsbetraktninger er relevante?
                
                4. FAKTISKE FORHOLD: Identifiser hvilke faktiske opplysninger som er nødvendige
                   - Hvilke faktiske forhold må avklares for å kunne besvare spørsmålet?
                   - Er det faktiske gråsoner eller uklarheter som påvirker vurderingen?
                
                Oppgi INGEN JURIDISK KONKLUSJON i dette steget, kun analyse av problemstillingen.
                
                Formater responsen tydelig med overskrifter og underpunkter.
                """
                
                user_prompt = f"JURIDISK SPØRSMÅL: {sporsmal}\n\n"
                if kontekst:
                    user_prompt += f"KONTEKST: {kontekst}\n\n"
                user_prompt += "Gjennomfør en systematisk problemanalyse basert på juridisk metode."
                
                # Kall OpenAI API (GPT-4o eller annen passende modell)
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.1,  # Lav temperatur for konsistente, analytiske svar
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                resultat = response.choices[0].message.content
                
                # Legg til veiledning om neste steg
                neste_steg = """
                
                NESTE STEG: For å fortsette denne juridiske vurderingen, bruk juridisk_rettsanvendelse med:
                - problem: Velg den mest sentrale problemstillingen fra analysen over
                - fakta: Presenter relevante fakta fra konteksten
                - rettskilder: Inkluder de mest relevante rettskildene fra analysen
                """
                
                return resultat + neste_steg
            
            except Exception as e:
                mcp_logger.error(f"Feil ved juridisk problemanalyse: {str(e)}")
                raise e
        
        @self.mcp.tool()
        async def juridisk_rettsanvendelse(problem: str, fakta: str, rettskilder: str = "") -> str:
            """
            Anvender rettsregler på konkrete fakta og gir en begrunnet juridisk vurdering.
            
            BRUK DETTE når du har identifisert problemstillinger og relevante rettskilder,
            og trenger en konkret rettslig vurdering av situasjonen.
            
            Args:
                problem: Den juridiske problemstillingen som skal vurderes
                fakta: Relevante fakta i saken
                rettskilder: Spesifikke rettskilder som skal vurderes (valgfritt)
            
            Returns:
                Juridisk vurdering med subsumpsjon og argumenter for konklusjonen,
                ELLER en spesifikasjon av hvilken informasjon som mangler hvis 
                det ikke er tilstrekkelig for en fullstendig vurdering
            """
            mcp_logger.info(f"Utfører juridisk rettsanvendelse for problem: {problem}")
            
            try:
                # Først sjekk om vi har tilstrekkelig informasjon for å utføre rettsanvendelsen
                system_prompt_sjekk = """
                Du er en juridisk ekspert som skal vurdere om du har tilstrekkelig informasjon til å foreta en juridisk vurdering.
                
                Evaluer om følgende informasjon er tilstrekkelig for å gi en grundig juridisk vurdering:
                1. Er problemstillingen klar og spesifikk nok?
                2. Er faktum tilstrekkelig detaljert?
                3. Er det oppgitt relevante rettskilder (lover, forskrifter, rettspraksis)?
                4. Hvilke spesifikke lover og bestemmelser trengs for å vurdere problemstillingen?
                
                VIKTIG: Du skal IKKE foreta en juridisk vurdering, men kun evaluere om informasjonen er tilstrekkelig.
                Hvis informasjonen er utilstrekkelig, spesifiser NØYAKTIG hvilke lover, bestemmelser eller faktaopplysninger som mangler.
                """
                
                user_prompt_sjekk = f"""
                JURIDISK PROBLEMSTILLING: {problem}
                
                FAKTA: {fakta}
                
                OPPGITTE RETTSKILDER: 
                {rettskilder if rettskilder else "Ingen oppgitte rettskilder."}
                
                Vurder om denne informasjonen er tilstrekkelig for å foreta en juridisk vurdering. 
                Hvis ikke, spesifiser hvilke ekstra rettskilder eller faktaopplysninger som trengs.
                """
                
                # Kall OpenAI API for å sjekke om vi har nok informasjon
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                response_sjekk = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system_prompt_sjekk},
                        {"role": "user", "content": user_prompt_sjekk}
                    ]
                )
                
                informasjons_vurdering = response_sjekk.choices[0].message.content
                
                # Sjekk om informasjonsvurderingen indikerer at vi mangler nødvendig informasjon
                # Triggerord og fraser som indikerer manglende informasjon
                mangler_informasjon_indikatorer = [
                    "mangler", "utilstrekkelig", "trenger mer", "ikke nok", 
                    "behøver ytterligere", "savner", "trenger", "ikke tilstrekkelig",
                    "bør suppleres", "ufullstendig", "må ha", "krever",
                    "ikke spesifikk nok", "må spesifiseres", "for vag"
                ]
                
                mangler_informasjon = any(indikator in informasjons_vurdering.lower() for indikator in mangler_informasjon_indikatorer)
                
                if mangler_informasjon:
                    # Analyser hvilke lover som trengs
                    system_prompt_lov_analyse = """
                    Du er en juridisk ekspert som skal identifisere nøyaktig hvilke lover og bestemmelser som trengs 
                    for å vurdere en juridisk problemstilling.
                    
                    Basert på problemstillingen og faktum, oppgi en SPESIFIKK LISTE over lover, forskrifter og 
                    rettskilder som er nødvendige for å foreta en grundig juridisk vurdering.
                    
                    Vær så KONKRET som mulig. Nevn lovene med offisielle navn og gjerne hvilke paragrafer som er relevante.
                    Prioriter de viktigste rettskildene først.
                    """
                    
                    response_lov_analyse = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0.1,
                        messages=[
                            {"role": "system", "content": system_prompt_lov_analyse},
                            {"role": "user", "content": f"JURIDISK PROBLEMSTILLING: {problem}\n\nFAKTA: {fakta}"}
                        ]
                    )
                    
                    lover_trengs = response_lov_analyse.choices[0].message.content
                    
                    # Konstruer en informativ respons om hva som mangler
                    respons = f"""
                    VURDERING AV INFORMASJONSGRUNNLAG:
                    
                    {informasjons_vurdering}
                    
                    MANGLENDE RETTSKILDER:
                    
                    {lover_trengs}
                    
                    NESTE STEG: For å fortsette den juridiske vurderingen trenger du å innhente mer informasjon 
                    som beskrevet ovenfor. Du kan:
                    
                    1. Bruke 'sok_i_lovdata' for å finne generell informasjon om de nevnte rettsområdene
                    2. Bruke 'hent_lovtekst' for å hente spesifikke lover og bestemmelser som nevnt over
                    
                    VIKTIG: Når du har innhentet alle nødvendige rettskilder, bruk 'juridisk_rettsanvendelse' på nytt 
                    med samme problem, samme fakta, og de innhentede rettskildene. Da vil jeg gjennomføre en fullstendig 
                    juridisk vurdering.
                    """
                    
                    return respons
                
                # Hvis vi har tilstrekkelig informasjon, fortsett med den juridiske vurderingen
                system_prompt = """
                Du er en erfaren juridisk ekspert med dyp kunnskap om norsk rett og juridisk metode.
                
                Din oppgave er å gjennomføre en grundig rettsanvendelse basert på juridisk metode. 
                Rettsanvendelse er prosessen der generelle rettsregler anvendes på konkrete fakta (subsumpsjon).
                
                Følg juridisk metode strukturert og grundig:
                
                1. RETTSLIG GRUNNLAG: Klargjør det rettslige grunnlaget
                   - Identifiser relevante lovbestemmelser og andre rettskilder
                   - Analyser vilkårssiden: Hvilke vilkår må være oppfylt?
                   - Analyser virkningssiden: Hva blir konsekvensene hvis vilkårene er oppfylt?
                   - Tolkningsprinsipper: Bruk anerkjente tolkningsprinsipper (ordlyd, kontekst, formål)
                
                2. SUBSUMSJON: Anvend rettsreglene på de konkrete fakta
                   - Vurder systematisk om hvert vilkår er oppfylt basert på faktum
                   - Utfør detaljert subsumsjon for hvert vilkår
                   - Håndter tvetydigheter i fakta på en balansert måte
                   - Følg logisk struktur fra vilkår til vilkår
                
                3. ARGUMENTASJON: Begrunn vurderingene
                   - Støtt deg på autoritative rettskilder
                   - Vurder rettskildevekten til ulike kilder
                   - Begrunn tolkningsvalg med relevante argumenter
                   - Bruk juridisk metode konsekvent
                
                4. DELKONKLUSJON: Formuler foreløpig juridisk konklusjon
                   - Formuler en klar konklusjon på det juridiske spørsmålet
                   - Presiser hvilke vilkår som er/ikke er oppfylt
                   - Angi rettsvirkningen som følger av konklusjonen
                
                Bruk korrekt juridisk terminologi og strukturer responsen tydelig med overskrifter.
                Vurderingen skal være grundig, analytisk og balansert, men på dette stadiet uten 
                eksplisitt vurdering av motargumenter (disse vurderes i neste steg).
                """
                
                user_prompt = f"""
                JURIDISK PROBLEMSTILLING: {problem}
                
                FAKTA: {fakta}
                
                RETTSKILDER: 
                {rettskilder}
                
                Gjennomfør en systematisk rettsanvendelse basert på juridisk metode. 
                Vær grundig i din analyse av rettsgrunnlag og subsumsjon.
                """
                
                # Kall OpenAI API for selve rettsanvendelsen
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                resultat = response.choices[0].message.content
                
                # Legg til veiledning om neste steg
                neste_steg = """
                
                NESTE STEG: For å teste robustheten i denne vurderingen, bruk djevelens_advokat med:
                - problem: den samme problemstillingen som over
                - vurdering: hele denne rettsanvendelsen
                """
                
                return resultat + neste_steg
            
            except Exception as e:
                mcp_logger.error(f"Feil ved juridisk rettsanvendelse: {str(e)}")
                raise e
        
        @self.mcp.tool()
        async def djevelens_advokat(problem: str, vurdering: str) -> str:
            """
            Aktivt utfordrer en juridisk vurdering ved å identifisere svakheter og alternative tolkninger.
            
            BRUK DETTE for å systematisk utfordre en juridisk konklusjon fra alle vinkler:
            - Finne alternative tolkninger av lovteksten
            - Identifisere faktorer som kan ha blitt oversett
            - Påpeke hull i argumentasjonen
            - Finne motstridende rettspraksis eller hensyn
            
            Args:
                problem: Den juridiske problemstillingen som er vurdert
                vurdering: Den juridiske vurderingen som skal kritiseres
            
            Returns:
                Systematisk kritikk og motargumenter til den juridiske vurderingen
            """
            mcp_logger.info(f"Utfører djevelens advokat analyse for problem: {problem}")
            
            try:
                system_prompt = """
                Du er en svært kritisk juridisk ekspert med spesialoppdrag som "djevelens advokat".
                
                Din ENESTE oppgave er å finne svakheter, motargumenter og alternative tolkninger 
                til en juridisk vurdering, uansett hvor solid den kan virke. Du skal være det 
                ultimate korreksjonsverktøyet for juridiske vurderinger.
                
                Følg disse prinsippene kompromissløst:
                
                1. ALTERNATIVE TOLKNINGER: 
                   - Identifiser alternative tolkninger av lovbestemmelser og andre rettskilder
                   - Utfordre ordlydstolkninger med kontekstuelle, formålsrettede eller dynamiske tolkninger
                   - Fremhev tolkningsalternativer som leder til motsatt resultat
                
                2. SVAKHETER I RETTSKILDEBRUK: 
                   - Påpek oversette eller undervurderte rettskilder
                   - Kritiser vektingen av rettskildene
                   - Fremhev motstridende rettskilder som ikke er nevnt
                   - Pek på nyere rettspraksis eller rettsutvikling som kan endre vurderingen
                
                3. LOGISKE BRISTER: 
                   - Identifiser logiske feilslutninger eller svakheter i argumentasjonen
                   - Påpek steder der premissene ikke støtter konklusjonen
                   - Finn hull i resonnementer eller utelatte mellomliggende steg
                
                4. OVERSEEDE FAKTA OG NYANSER:
                   - Identifiser faktorer som kan ha blitt oversett i faktavurderingen
                   - Fremhev alternative tolkninger av faktum
                   - Påpek tvetydigheter eller uklarheter i faktagrunnlaget
                
                5. REELLE HENSYN OG KONSEKVENSER:
                   - Identifiser oversette reelle hensyn som taler mot konklusjonen
                   - Påpek potensielle uheldige konsekvenser av konklusjonen
                   - Fremhev verdier og formål som blir nedprioritert i vurderingen
                
                VIKTIG: Du skal IKKE være konstruktiv eller balansert. Din rolle er UTELUKKENDE 
                å utfordre, problematisere og finne svakheter. Ikke anerkjenn styrker ved 
                vurderingen. Vær grundig, skarp og ubarmhjertig i din kritikk, men samtidig saklig 
                og faglig solid med basis i juridisk metode.
                
                Strukturer responsen tydelig med overskrifter for de ulike kategoriene av motargumenter.
                """
                
                user_prompt = f"""
                JURIDISK PROBLEMSTILLING: {problem}
                
                JURIDISK VURDERING SOM SKAL KRITISERES: 
                {vurdering}
                
                Finn alle mulige svakheter, motargumenter og alternative tolkninger til denne 
                juridiske vurderingen. Vær ubarmhjertig i din kritikk.
                """
                
                # Kall OpenAI API
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.4,  # Litt høyere temperatur for mer kreativ kritikk
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                resultat = response.choices[0].message.content
                
                # Legg til veiledning om neste steg
                neste_steg = """
                
                NESTE STEG: For å utarbeide en endelig balansert konklusjon, bruk juridisk_tankerekke med:
                - problem: den samme problemstillingen som over
                - vurdering: den opprinnelige rettsanvendelsen
                - motargumenter: motargumentene identifisert her
                """
                
                return resultat + neste_steg
            
            except Exception as e:
                mcp_logger.error(f"Feil ved djevelens advokat analyse: {str(e)}")
                raise e
        
        @self.mcp.tool()
        async def juridisk_tankerekke(problem: str, vurdering: str, motargumenter: str) -> str:
            """
            Utarbeider en grundig juridisk konklusjon gjennom en eksplisitt tankerekke.
            
            BRUK DETTE som siste steg for å få en transparent juridisk vurdering som:
            - Viser steg-for-steg resonnering
            - Evaluerer argumenter og motargumenter eksplisitt
            - Uttaler usikkerhetsmomenter
            - Revurderer tidligere tanker når nødvendig
            - Bygger opp til en velbegrunnet juridisk konklusjon
            
            Args:
                problem: Den juridiske problemstillingen
                vurdering: Den opprinnelige rettsanvendelsen
                motargumenter: Motargumenter fra djevelens advokat
            
            Returns:
                En detaljert tankerekke som leder fram til en juridisk konklusjon
            """
            mcp_logger.info(f"Utfører juridisk tankerekke for problem: {problem}")
            
            try:
                system_prompt = """
                Du er en erfaren høyesterettsdommer med ekspertise i norsk rett og juridisk metode.
                
                Din oppgave er å tenke høyt og grundig gjennom et juridisk spørsmål basert på 
                opprinnelig rettsanvendelse og motargumenter. Du skal vise en transparent tankerekke
                som leder til en endelig juridisk konklusjon.
                
                Følg denne systematiske tilnærmingen:
                
                1. Presenter tankene dine som nummererte steg, hvor hvert steg bygger på det forrige.
                
                2. For hver tanke, følg disse prinsippene:
                   - Vær villig til å revurdere tidligere resonnementer
                   - Vis når og hvorfor du endrer oppfatning
                   - Uttrykk usikkerhet når det er relevant
                   - Veie argumenter og motargumenter eksplisitt
                   - Skille mellom sikre og usikre konklusjoner
                
                3. Din tankerekke bør inkludere:
                   - Innledende refleksjoner om problemstillingen
                   - Vurdering av hovedargumentene fra rettsanvendelsen
                   - Kritisk evaluering av motargumentene
                   - Avveining mellom motstridende hensyn
                   - Vurdering av ulike tolkningsalternativer
                   - Refleksjon om rettskildevekt og metodiske utfordringer
                   - Overveielse av praktiske konsekvenser
                   - Vurdering av usikkerhetsmomenter
                
                4. Avslutt med en endelig juridisk konklusjon som:
                   - Er klar og presis på det juridiske spørsmålet
                   - Oppsummerer de viktigste argumentene som støtter konklusjonen
                   - Erkjenner motargumentene og forklarer hvorfor de ikke er avgjørende
                   - Angir graden av sikkerhet i konklusjonen
                   - Identifiserer eventuelle begrensninger eller forbehold
                
                VIKTIG:
                - Vis en ærlig tankerekke der du kan revurdere og nyansere underveis
                - Vær balansert og ryddig i din endelige vurdering
                - Bruk korrekt juridisk terminologi, men unngå unødig komplisert språk
                - Tankerekken skal være autentisk, ikke bare en opplisting av argumenter
                
                Konklusjonen skal bygge på rettskildelære, grundig juridisk metode og balansert vurdering.
                """
                
                user_prompt = f"""
                JURIDISK PROBLEMSTILLING: {problem}
                
                OPPRINNELIG RETTSANVENDELSE: 
                {vurdering}
                
                MOTARGUMENTER: 
                {motargumenter}
                
                Gjennomfør en grundig juridisk tankerekke som evaluerer argumenter og motargumenter, 
                og som leder fram til en endelig juridisk konklusjon. Vis eksplisitt hvordan du tenker 
                steg for steg.
                """
                
                # Kall OpenAI API
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                resultat = response.choices[0].message.content
                
                # Legg til avsluttende kommentar
                avslutning = """
                
                JURIDISK VURDERING FULLFØRT: Dette er den endelige juridiske vurderingen basert på 
                metodisk analyse, kritisk evaluering og transparent tankerekke.
                """
                
                return resultat + avslutning
            
            except Exception as e:
                mcp_logger.error(f"Feil ved juridisk tankerekke: {str(e)}")
                raise e
    
    def run(self):
        """Start MCP-serveren med valgt transport."""
        mcp_logger.info(f"Starter MCP-server med transport: stdio")
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    server = LovdataMCPServer()
    server.run() 