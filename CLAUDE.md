# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Principles

**Philosophy 1: Do the simplest thing that works.**

**Philosophy 2: Make zero assumptions or decisions about system design and implementaiton at any level of the project. Ask the user for clarifiaciton and context.**

### Simplicity & Conciseness

- Write the minimum code needed to solve the problem correctly — no more
- Delete code that isn't doing meaningful work; dead code is a liability
- Prefer a clear 10-line function over an abstracted 50-line one
- No speculative abstractions: don't build for hypothetical future requirements
- Create/Edit the least number of files necessary for each task

### Logging

Log only where execution is hard to trace: proxy failures, unexpected branches, and operation outcomes. Avoid logging in straightforward code paths.

- `DEBUG` — tricky request-level details
- `INFO` — top-level operation start/finish
- `WARNING` — recoverable failures (retries, rotations)
- `ERROR` — unrecoverable failures

### What to Avoid

- Commented-out code (delete it; git has history)
- Wrapper functions that only call one other function with no added logic
- Generic exception handling that silently swallows errors (`except Exception: pass`)

/Users/luisbarajas/Desktop/Projects/Resume-Job-Matching-System/offline-data-pipeline is a sibling directory under the same project as this one. Use it as context for the offline pipeline when necessary.

/Users/luisbarajas/Desktop/Projects/Resume-Job-Match is an older verion of this project. Ignore it completely.

## Project Overview

Real-time REST API that accepts a resume and returns a ranked list of job matches with explanations. It is the online counterpart to the offline data pipeline, which handles job collection, preprocessing, embedding, and ChromaDB indexing. This service is read-only against the index — it never rebuilds or writes to ChromaDB.

## Architecture

### Module Layout

- `config.py` — API keys (`VOYAGE_API_KEY`, `COHERE_API_KEY`, `OPENROUTER_API_KEY`), paths (`CHROMA_DIR`, `CHROMA_COLLECTION`), model names (`VOYAGE_MODEL`, `COHERE_RERANK_MODEL`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`), thresholds (`TOP_K_RETRIEVE=100`, `TOP_K_RERANK_DEFAULT=10`, `CHROMA_EF_SEARCH=200`)
- `app/main.py` — FastAPI app entry point; mounts router, starts Uvicorn
- `app/routes.py` — `POST /match` endpoint; validates input, calls pipeline steps in order, returns JSON array; also `GET /health` (liveness) and `GET /ready` (checks VoyageAI, ChromaDB, OpenRouter)
- `app/schemas.py` — Pydantic models: `MatchRequest` (resume, top_k, education, years_of_experience) and `MatchResult` (job_id, title, company, url, score, explanation, corpus_warning)
- `pipeline/embed.py` — `embed(text) → list[float]`; calls VoyageAI with `input_type="query"` (asymmetric to offline `"document"` embeddings — required for correct retrieval)
- `pipeline/retrieve.py` — `retrieve(vector, top_k, filters) → list[dict]`; opens `PersistentClient` on `CHROMA_DIR`, queries `"job_descriptions"` collection; applies optional metadata filters
- `pipeline/rerank.py` — `rerank(resume, candidates, top_k) → list[dict]`; Cohere `rerank-english-v3.0`; document string includes title, company, responsibilities, and qualifications
- `pipeline/generate.py` — `explain(resume, job) → tuple[str, bool]`; single LangChain prompt via OpenRouter/DeepSeek; returns `(explanation_text, corpus_warning)`

### Inference Flow

1. Validate input (Pydantic)
2. Embed resume (`input_type="query"`)
3. Retrieve top-100 from ChromaDB (with optional metadata filters)
4. Rerank to top-k (Cohere cross-encoder)
5. Per job: `explain()` — single LLM call returns explanation text + corpus_warning flag
6. Return JSON array

Full spec: `context/pipeline-architecture.md`

### Key Constraints

- Never embed the resume with `input_type="document"` — it must be `"query"` (asymmetric retrieval)
- `responsibilities` and `qualifications` from ChromaDB are JSON strings — always `json.loads()` before use
- `max_yoe == -1` means unknown; skip the filter for that job. When set, it is the minimum YOE floor required — filter with `$lte` to find roles at or below the user's experience level.
- `min_education == ""` means unknown; treat as no constraint in filters
- Do not rebuild or write to the ChromaDB index — read-only

## Development Commands

```bash
# Start the API server (hot reload)
uvicorn app.main:app --reload

# Run all tests
pytest
```

## Environment Variables

| Variable             | Required | Default        | Purpose                                             |
| -------------------- | -------- | -------------- | --------------------------------------------------- |
| `VOYAGE_API_KEY`     | Yes      | —              | VoyageAI resume embedding                           |
| `COHERE_API_KEY`     | Yes      | —              | Cohere reranking                                    |
| `OPENROUTER_API_KEY` | Yes      | —              | DeepSeek explanation generation                     |
| `CHROMA_PATH`        | No       | `chroma_index` | Path to ChromaDB index pulled from offline pipeline |

## Data Source

ChromaDB index and SQLite schema are produced by the offline pipeline and committed to its repo every 6 hours. See `context/offline-pipeline-context.md` for the full schema and artifact contracts.
