import json
import logging

import cohere

from config import COHERE_API_KEY, COHERE_RERANK_MODEL, TOP_K_RERANK_DEFAULT

logger = logging.getLogger(__name__)


def _format_job(candidate: dict) -> str:
    parts = [f"{candidate.get('title', '')} at {candidate.get('company', '')}"]

    for section, label in [("responsibilities", "Responsibilities"), ("qualifications", "Qualifications")]:
        try:
            items = json.loads(candidate.get(section, "[]"))
        except (json.JSONDecodeError, TypeError):
            items = []
        if items:
            parts.append(f"\n{label}:\n" + "\n".join(f"- {item}" for item in items))

    return "\n".join(parts)


def rerank(resume: str, candidates: list[dict], top_k: int = TOP_K_RERANK_DEFAULT) -> list[dict]:
    logger.info("Reranking %d candidates to top-%d", len(candidates), top_k)
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    documents = [_format_job(c) for c in candidates]

    results = co.rerank(model=COHERE_RERANK_MODEL, query=resume, documents=documents, top_n=top_k)

    reranked = []
    for r in results.results:
        candidate = dict(candidates[r.index])
        candidate["score"] = r.relevance_score
        reranked.append(candidate)

    logger.info("Reranking complete, returning %d results", len(reranked))
    return reranked
