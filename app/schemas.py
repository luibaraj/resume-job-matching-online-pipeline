from typing import Optional

from pydantic import BaseModel

from config import TOP_K_RERANK_DEFAULT


class MatchRequest(BaseModel):
    resume: str
    top_k: int = TOP_K_RERANK_DEFAULT
    explain_top_k: Optional[int] = None
    education: Optional[str] = None
    years_of_experience: Optional[int] = None
    internship: Optional[bool] = None


class MatchResult(BaseModel):
    job_id: str
    title: str
    company: str
    url: str
    score: float
    explanation: str
    corpus_warning: bool
