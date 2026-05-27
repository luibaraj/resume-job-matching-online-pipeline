from typing import Optional

from pydantic import BaseModel

from config import TOP_K_RERANK_DEFAULT


class MatchRequest(BaseModel):
    resume: str
    top_k: int = TOP_K_RERANK_DEFAULT
    education: Optional[str] = None
    years_of_experience: Optional[int] = None


class MatchResult(BaseModel):
    job_id: str
    title: str
    company: str
    url: str
    score: float
    explanation: str
    corpus_warning: bool
