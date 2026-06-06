import json
import logging
import random
import re

import cohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    TOP_K_RERANK_DEFAULT,
)

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 10
_WINDOW_STEP = 5

_RANK_PROMPT = """\
You are a job ranking assistant. Rank the following roles relative to one another — not each scored in isolation. \
The goal is to identify which role is the best use of the candidate's time to pursue, given their background.

Rank roles LOWER on the list if they:
- Require a security clearance
- Appear to be posted by a staffing agency or recruiter (not the employer directly)

Apply these criteria based on whatever context appears in the job title, responsibilities, and qualifications.

Output ONLY a permutation in this exact format, from best to worst fit: [0] > [1] > [2] > ...
Use only the job numbers provided. Output nothing else.

Resume:
{resume}

Jobs:
{jobs}"""


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


def _parse_permutation(text: str, window_size: int) -> list[int] | None:
    indices = [int(m) for m in re.findall(r"\[(\d+)\]", text)]
    if sorted(indices) == list(range(window_size)):
        return indices
    return None


def _rerank_llm(resume: str, candidates: list[dict], top_k: int) -> list[dict]:
    logger.info("LLM sliding-window reranking %d candidates to top-%d", len(candidates), top_k)
    ranked = list(candidates)
    n = len(ranked)

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )
    chain = ChatPromptTemplate.from_messages([("user", _RANK_PROMPT)]) | llm | StrOutputParser()

    start = max(0, n - _WINDOW_SIZE)
    while True:
        end = min(start + _WINDOW_SIZE, n)
        window = ranked[start:end]
        actual_size = len(window)

        shuffle_order = list(range(actual_size))
        random.shuffle(shuffle_order)
        shuffled = [window[i] for i in shuffle_order]

        jobs_text = "\n\n".join(f"[{i}] {_format_job(shuffled[i])}" for i in range(actual_size))

        try:
            output = chain.invoke({"resume": resume, "jobs": jobs_text})
            permutation = _parse_permutation(output, actual_size)
        except Exception:
            logger.warning("LLM call failed during sliding window rerank at window start=%d", start)
            permutation = None

        if permutation is not None:
            # permutation[i] = index in shuffled that should go to position i
            reordered = [shuffled[permutation[i]] for i in range(actual_size)]
            ranked[start:end] = reordered

        if start == 0:
            break
        start = max(0, start - _WINDOW_STEP)

    for i, candidate in enumerate(ranked):
        candidate = dict(candidate)
        candidate["score"] = (n - i) / n
        ranked[i] = candidate

    logger.info("LLM reranking complete, returning %d results", top_k)
    return ranked[:top_k]


def rerank(resume: str, candidates: list[dict], top_k: int = TOP_K_RERANK_DEFAULT, method: str = "cohere") -> list[dict]:
    if method == "llm":
        return _rerank_llm(resume, candidates, top_k)

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
