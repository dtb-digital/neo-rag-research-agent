# Lovdata Ingest Script

Dette scriptet lar deg laste inn lovtekster fra en URL til en Pinecone vektorbase for bruk med RAG-agenten.

## Avhengigheter

Installer alle nødvendige avhengigheter:

```bash
pip install -r scripts/requirements-ingest.txt
```

## Miljøvariabler

Følgende miljøvariabler må være satt:

```bash
export OPENAI_API_KEY=din_openai_api_nøkkel
export PINECONE_API_KEY=din_pinecone_api_nøkkel
export PINECONE_ENVIRONMENT=din_pinecone_miljø  # f.eks. "gcp-starter"
```

## Bruk

Kjør scriptet med en URL til en lovtekst:

```bash
python scripts/ingest_lovdata.py --url https://lovdata.no/dokument/NL/lov/1814-05-17-nn
```

### Tilgjengelige kommandolinjeargumenter:

- `--url` (påkrevd): URL til lovdokumentet
- `--index-name` (valgfri): Navn på Pinecone-indeksen (standard: "lovdata-index")
- `--batch-size` (valgfri): Antall dokumenter per batch for opplasting (standard: 100)

## Hvordan det fungerer

1. Henter HTML-innhold fra den angitte URL-en
2. Renser HTML og ekstraherer lovteksten
3. Bruker LLM (OpenAI) for å analysere og ekstrahere strukturert metadata fra teksten
4. Deler opp teksten i mindre chunks for vektorisering
5. Genererer embeddings for hver chunk ved hjelp av OpenAI embeddings-modell
6. Laster opp til Pinecone vektorbasen med metadata

## Eksempler på lovdata-URL-er som kan brukes:

- Grunnloven (bokmål): https://lovdata.no/dokument/NL/lov/1814-05-17
- Grunnloven (nynorsk): https://lovdata.no/dokument/NL/lov/1814-05-17-nn
- Straffeloven: https://lovdata.no/dokument/NL/lov/2005-05-20-28
- Arbeidsmiljøloven: https://lovdata.no/dokument/NL/lov/2005-06-17-62

## Feilsøking

Hvis du opplever problemer:

1. Sjekk at alle miljøvariabler er korrekt satt
2. Verifiser at URL-en er gyldig og peker til en lovdata-side
3. Sjekk om Pinecone-indeksen eksisterer og at du har tilgang til den

Hvis du får feil relatert til "rate limits" fra OpenAI eller Pinecone, prøv å redusere batch-størrelsen eller legg inn pauser mellom API-kall. 