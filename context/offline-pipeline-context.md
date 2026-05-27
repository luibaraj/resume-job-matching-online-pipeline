# Offline Pipeline — Context & Contracts

**Repo:** `Resume-Job-Matching-System/offline-data-pipeline`

This document describes what the offline pipeline does, what it produces, and the exact schemas this online pipeline depends on.

---

## Overview

The offline pipeline builds a structured, semantically-indexed job corpus. It runs every 6 hours via GitHub Actions and commits its two output artifacts back to the repo:

- `jobs.db` — SQLite database of scraped and processed jobs
- `chroma_index/` — ChromaDB HNSW vector index (cosine, 1024-dim)

**Full pipeline:** `scrape → preprocess → extract → extract-meta → embed → index`

Jobs older than 2 days are purged from SQLite before each run. Vectors older than 24 hours are pruned from ChromaDB during the index step.

---

## Data Source

**Source:** LinkedIn only, via the JobSpy library.

**Search term pools:**
- ML/AI: Machine Learning Engineer, ML Engineer, ML/AI Engineer, AI Engineer, Deep Learning Engineer, NLP Engineer, Research Engineer
- Data: Data Scientist, Customer Data Scientist, Applied Scientist, Machine Learning Scientist, Data Science Engineer, Data Analyst

**Volume:** ~300 jobs/day. Only postings from the last 6 hours are fetched each run.

---

## Pipeline Steps

### 1. `scrape`
Fetches jobs from LinkedIn and writes raw rows to SQLite.

**Writes to `jobs` table:** `id`, `title`, `company`, `location`, `job_url`, `description`, `date_posted`, `scraped_at`

### 2. `preprocess`
Cleans raw job descriptions.

**Reads:** `description`  
**Writes:** `description_clean` — unicode normalization, markdown stripping, whitespace collapse

### 3. `extract`
Uses DeepSeek LLM (via OpenRouter) to parse structured fields from the cleaned description. Runs with a 10-worker thread pool.

**Reads:** `description_clean`  
**Writes:**
- `qualifications` — JSON-serialized list of qualification strings
- `responsibilities` — JSON-serialized list of responsibility strings

### 4. `extract-meta`
Parses education and experience requirements from qualifications text using regex + LLM.

**Reads:** `qualifications`  
**Writes:**
- `min_education` — str: `"BS"`, `"MS"`, `"PhD"`, or `""` (unknown)
- `max_yoe` — int: maximum years of experience required, or `-1` (unknown)

### 5. `embed`
Generates 1024-dim embeddings via VoyageAI.

**Reads:** `description_clean`  
**Writes:** `jd_embedding` — BLOB (1024 × float32, ~4 KB per row)  
**Model:** `voyage-3.5-lite`, `input_type="document"`, batch size 128

### 6. `index`
Loads all embedded jobs into ChromaDB and prunes stale vectors.

**Reads:** all rows where `jd_embedding IS NOT NULL`  
**Writes:** ChromaDB collection `"job_descriptions"`  
**Prunes:** vectors with `scraped_at` older than 24 hours

---

## SQLite Schema

**Table: `jobs`**

Base columns (created at init):

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | LinkedIn job ID |
| `title` | TEXT | |
| `company` | TEXT | |
| `location` | TEXT | |
| `job_url` | TEXT UNIQUE | LinkedIn URL |
| `description` | TEXT | Raw text from LinkedIn |
| `date_posted` | TEXT | |
| `scraped_at` | TEXT | ISO datetime, default `datetime('now')` |

Dynamically added columns (via ALTER TABLE):

| Column | Type | Notes |
|---|---|---|
| `description_clean` | TEXT | Normalized description |
| `qualifications` | TEXT | JSON list |
| `responsibilities` | TEXT | JSON list |
| `max_yoe` | INTEGER | `-1` = unknown |
| `min_education` | TEXT | `""` = unknown |
| `jd_embedding` | BLOB | 1024-dim float32 binary |

---

## ChromaDB Schema

**Collection:** `"job_descriptions"`  
**Distance metric:** Cosine (`hnsw:space = "cosine"`)  
**HNSW construction ef:** 200  
**IDs:** job `id` as string  
**Embeddings:** 1024-dim float32 (pre-computed, no embedding function on the collection)

**Metadata per document:**

| Key | Type | Sentinel | Notes |
|---|---|---|---|
| `title` | str | `""` | Job title |
| `company` | str | `""` | Company name |
| `job_url` | str | `""` | LinkedIn posting URL |
| `max_yoe` | int | `-1` | Max years of experience required |
| `min_education` | str | `""` | `"BS"`, `"MS"`, or `"PhD"` |
| `responsibilities` | str | `"[]"` | JSON-serialized list |
| `qualifications` | str | `"[]"` | JSON-serialized list |

---

## Models & Environment

| Setting | Value |
|---|---|
| Embedding model | `voyage-3.5-lite` (VoyageAI) |
| Embedding dimension | 1024 |
| Extraction LLM | `deepseek/deepseek-chat-v3-0324` (OpenRouter) |
| `CHROMA_PATH` | env var, default `"chroma_index"` |
| `DB_PATH` | env var, default `"jobs.db"` |

---

## How This Online Pipeline Consumes the Artifacts

- Pull `chroma_index/` from the offline pipeline's git repo; set as `CHROMA_PATH`
- Open the collection via `chromadb.PersistentClient(path=CHROMA_PATH).get_collection("job_descriptions")`
- `responsibilities` and `qualifications` are JSON strings — deserialize with `json.loads()` before use
- `max_yoe == -1` means no experience requirement is stated; treat as no upper bound when filtering
- `min_education == ""` means no education requirement is stated; treat as no constraint when filtering
- Resume embeddings must use `input_type="query"` (asymmetric to the offline `"document"` embeddings)
