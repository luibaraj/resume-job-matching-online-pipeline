import logging

import chromadb
import voyageai
from fastapi import APIRouter, HTTPException
from openai import OpenAI

from app.schemas import MatchRequest, MatchResult
from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COHERE_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    TOP_K_RETRIEVE,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)
from pipeline.embed import embed
from pipeline.generate import explain
from pipeline.rerank import rerank
from pipeline.retrieve import retrieve

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_filters(education: str | None, years_of_experience: int | None) -> dict | None:
    clauses = []
    if years_of_experience is not None:
        clauses.append({"$and": [{"max_yoe": {"$ne": -1}}, {"max_yoe": {"$lte": years_of_experience}}]})
    if education is not None:
        clauses.append({"$and": [{"min_education": {"$ne": ""}}, {"min_education": {"$eq": education}}]})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@router.post("/match", response_model=list[MatchResult])
def match(req: MatchRequest):
    filters = _build_filters(req.education, req.years_of_experience)

    query_vector = embed(req.resume)
    candidates = retrieve(query_vector, top_k=TOP_K_RETRIEVE, filters=filters)
    if not candidates:
        return []

    ranked = rerank(req.resume, candidates, top_k=req.top_k)

    results = []
    for job in ranked:
        explanation_text, corpus_warning = explain(req.resume, job)
        results.append(MatchResult(
            job_id=job["job_id"],
            title=job.get("title", ""),
            company=job.get("company", ""),
            url=job.get("job_url", ""),
            score=job["score"],
            explanation=explanation_text,
            corpus_warning=corpus_warning,
        ))
    return results


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    checks = {}

    try:
        client = voyageai.Client(api_key=VOYAGE_API_KEY)
        client.embed(["ping"], model=VOYAGE_MODEL, input_type="query")
        checks["voyage"] = "ok"
    except Exception as e:
        logger.warning("VoyageAI readiness check failed: %s", e)
        checks["voyage"] = "error"

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_client.get_collection(CHROMA_COLLECTION)
        checks["chromadb"] = "ok"
    except Exception as e:
        logger.warning("ChromaDB readiness check failed: %s", e)
        checks["chromadb"] = "error"

    try:
        openai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        openai_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        checks["openrouter"] = "ok"
    except Exception as e:
        logger.warning("OpenRouter readiness check failed: %s", e)
        checks["openrouter"] = "error"

    if any(v != "ok" for v in checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return checks
