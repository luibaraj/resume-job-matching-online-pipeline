# Online Inference Pipeline — Architecture

## Purpose

This service receives a resume and returns a ranked list of job matches with explanations. It is the online counterpart to the offline data pipeline, which handles job collection, preprocessing, embedding, and ChromaDB indexing.

The ChromaDB index is built and committed to the offline pipeline's git repo on every run. This service pulls from that repo and queries the index directly — it never rebuilds the index.

---

## Data Source: Offline Pipeline

**Repo:** `Resume-Job-Matching-System/offline-data-pipeline`

**ChromaDB collection:** `"job_descriptions"`  
**Distance metric:** Cosine (HNSW)  
**Index path:** `chroma_index/` (env var `CHROMA_PATH`, default `"chroma_index"`)

**Metadata stored per job:**

| Field | Type | Notes |
|---|---|---|
| `title` | str | Job title |
| `company` | str | Company name |
| `job_url` | str | Link to posting |
| `max_yoe` | int | Max years of experience required; `-1` = unknown |
| `min_education` | str | Minimum education level; `""` = unknown |
| `responsibilities` | str | JSON-serialized list |
| `qualifications` | str | JSON-serialized list |

**Embedding model:** VoyageAI `voyage-3.5-lite`, 1024-dim float32  
**Update cadence:** Every 6 hours via GitHub Actions; `chroma_index/` committed back to git after each run.

---

## Inference Pipeline — 9 Steps

### 1. Input Validation
FastAPI receives `POST /match`. Pydantic validates:
- `resume`: string, minimum 50 characters
- `top_k`: integer, 1–50
- `education`: optional string (matches `min_education` metadata)
- `years_of_experience`: optional integer (compared against `max_yoe`)

### 2. Resume Embedding
The resume text is sent to VoyageAI with `input_type="query"` using model `voyage-3.5-lite`. Returns a 1024-dim float32 vector. Using `input_type="query"` (vs. `"document"`) is intentional — it optimizes the vector for similarity search against document-type embeddings stored in the index.

### 3. Vector Retrieval
The query vector hits the ChromaDB HNSW index. Top-100 candidates are retrieved by cosine similarity. Optional metadata filters applied inside the query:
- `education` → `where min_education == education`
- `years_of_experience` → `where max_yoe == -1 OR max_yoe >= years_of_experience`

Retrieving 100 candidates before reranking ensures the reranker has enough signal to work with.

### 4. Reranking
The top-100 candidates are sent to Cohere `rerank-english-v3.0`. Each candidate is represented as `"{title} at {company}"`. The cross-encoder scores each job against the full resume text jointly (not in isolation), producing a relevance score that captures semantic fit beyond embedding similarity. Returns top-k by relevance score.

### 5. Requirement Extraction
For each of the top-k jobs, an LLM (DeepSeek via OpenRouter) reads the job's `qualifications` and `responsibilities` and extracts:
- A list of required skills
- The job domain

### 6. Skill Matching
The LLM compares skills mentioned in the resume against the extracted job requirements. Produces a list of candidate skill match strings (claimed overlaps).

### 7. Hallucination Filtering
Each claimed skill match is verified as a case-insensitive substring of the job's raw qualifications/responsibilities text. Any match not literally present in the source text is dropped. This step is pure Python — no LLM, no API call.

**Rationale:** LLMs occasionally infer or paraphrase skills that aren't actually stated in the job description. Substring verification grounds every match in the source text, preventing the explanation from fabricating requirements.

### 8. Explanation Generation
The LLM summarizes the verified matches into a plain-English explanation of why the candidate fits the role. If no verified matches exist or the LLM call fails, the explanation falls back to the first 500 characters of the job description.

### 9. Response
Returns a JSON array. Each element:

```json
{
  "job_id": "string",
  "title": "string",
  "company": "string",
  "url": "string",
  "score": 0.91,
  "explanation": "string",
  "corpus_warning": false
}
```

`corpus_warning: true` means no verified skill matches were found — the explanation is a fallback and the match quality is uncertain.

---

## Tech Stack

| Component | Choice |
|---|---|
| API framework | FastAPI |
| Embedding | VoyageAI `voyage-3.5-lite` |
| Vector store | ChromaDB (persistent, pulled from offline pipeline) |
| Reranker | Cohere `rerank-english-v3.0` |
| LLM | DeepSeek `deepseek-chat-v3-0324` via OpenRouter |
| LLM SDK | `openai` Python SDK (pointed at OpenRouter base URL) |

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `VOYAGE_API_KEY` | Yes | — | VoyageAI embedding |
| `COHERE_API_KEY` | Yes | — | Cohere reranking |
| `OPENROUTER_API_KEY` | Yes | — | DeepSeek via OpenRouter |
| `CHROMA_PATH` | No | `chroma_index` | Path to ChromaDB index |
