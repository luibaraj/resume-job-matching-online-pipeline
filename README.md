# Resume Job Matching — Online Pipeline

REST API that accepts a resume and returns a ranked list of job matches with explanations.

## Overview

This service is the online counterpart to the [offline data pipeline](#related). It reads from a pre-built ChromaDB index of job descriptions (never writes to it) and runs a 4-stage ML pipeline: semantic embedding → vector retrieval → cross-encoder reranking → LLM explanation generation.

## Architecture

**Inference flow:**

1. Embed resume with VoyageAI (`input_type="query"`)
2. Retrieve top-100 candidates from ChromaDB (optional metadata filters)
3. Rerank to top-k with Cohere cross-encoder
4. Generate a 2–3 sentence match explanation per job (DeepSeek via OpenRouter)

**Modules:**

| File | Responsibility |
|------|---------------|
| `app/main.py` | FastAPI entry point |
| `app/routes.py` | `POST /match`, `GET /health`, `GET /ready` |
| `app/schemas.py` | `MatchRequest` and `MatchResult` Pydantic models |
| `config.py` | API keys, paths, model names, thresholds |
| `pipeline/embed.py` | Resume → embedding vector |
| `pipeline/retrieve.py` | Vector → ChromaDB candidates |
| `pipeline/rerank.py` | Candidates → reranked list |
| `pipeline/generate.py` | Job + resume → explanation |

## Prerequisites

- Python 3.10+
- A `chroma_index/` directory produced by the offline pipeline (see [Related](#related))

## Setup

```bash
git clone <repo-url>
cd resume-job-matching-online-pipeline

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys (see [Environment Variables](#environment-variables) below).

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `VOYAGE_API_KEY` | Yes | — | VoyageAI resume embedding |
| `COHERE_API_KEY` | Yes | — | Cohere reranking |
| `OPENROUTER_API_KEY` | Yes | — | DeepSeek explanation generation |
| `CHROMA_PATH` | No | `chroma_index` | Path to ChromaDB index from offline pipeline |

## Running

```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`.

## API Reference

### `POST /match`

Match a resume against the job index.

**Request body:**

```json
{
  "resume": "string (required)",
  "top_k": 10,
  "education": "Bachelor's",
  "years_of_experience": 3
}
```

`education` and `years_of_experience` are optional metadata filters.

**Response:** array of match objects

```json
[
  {
    "job_id": "string",
    "title": "string",
    "company": "string",
    "url": "string",
    "score": 0.95,
    "explanation": "string",
    "corpus_warning": false
  }
]
```

`corpus_warning: true` means the explanation may be unreliable due to limited job data.

### `GET /health`

Returns `{"status": "ok"}`.

### `GET /ready`

Checks connectivity to VoyageAI, ChromaDB, and OpenRouter. Returns readiness status.

## Testing

```bash
pytest
```

## Related

- [offline-data-pipeline](../offline-data-pipeline) — Scrapes LinkedIn (~300 jobs/day), embeds job descriptions, and writes the ChromaDB index this service reads from. Runs every 6 hours via GitHub Actions.
