# Neo RAG Research Agent

Dette er en RAG (Retrieval Augmented Generation) agent som bruker [LangGraph](https://github.com/langchain-ai/langgraph) for å besvare spørsmål basert på dokumenter lagret i Pinecone.

## Hva gjør den?

Prosjektet består av en "retrieval" graf (`src/retrieval_graph/graph.py`) med en "researcher" undergraf (`src/retrieval_graph/researcher_graph/graph.py`).

Retrieval-grafen håndterer chathistorikk og genererer svar basert på hentede dokumenter. Spesifikt:

1. Tar imot et bruker-spørsmål som input
2. Analyserer spørsmålet og bestemmer hvordan det skal rutes:
   - hvis spørsmålet er om lovverk, lager den en forskningsplan og sender den til researcher-grafen
   - hvis spørsmålet er uklart, ber den om mer informasjon
   - hvis spørsmålet er generelt, informerer den brukeren
3. Hvis spørsmålet handler om lovverk, kjører researcher-grafen for hvert steg i forskningsplanen:
   - genererer først en liste med søkestrenger basert på steget
   - henter deretter relevante dokumenter parallelt for alle søkestrenger
4. Til slutt genererer retrieval-grafen et svar basert på de hentede dokumentene og samtalehistorikken

## Komme i gang

1. Installer avhengigheter:
```bash
pip install -e .
```

2. Sett opp miljøvariabler i `.env`:
```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=lovdata-paragraf-test
```

## Utvikling

Under utvikling kan du redigere tidligere tilstander og kjøre appen på nytt fra tidligere tilstander for å debugge spesifikke noder. Lokale endringer vil automatisk bli anvendt via hot reload.

Du kan finne den nyeste dokumentasjonen om [LangGraph](https://github.com/langchain-ai/langgraph) her, inkludert eksempler og andre referanser.

LangGraph Studio integrerer også med [LangSmith](https://smith.langchain.com/) for mer detaljert sporing og samarbeid med teammedlemmer.
