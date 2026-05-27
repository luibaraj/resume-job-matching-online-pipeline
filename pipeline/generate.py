import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

_CORPUS_WARNING = "The corpus of jobs seems limited. The recommendation may be inaccurate."

_PROMPT = """\
You are a job fit evaluator. Given a candidate's resume and a job posting, do one of two things:

1. If the candidate is a good fit, write 2-3 sentences explaining why — be specific about matching skills and experience.
2. If the candidate is NOT a good fit, respond with exactly: "{corpus_warning}"

Job Title: {{title}}
Company: {{company}}

Responsibilities:
{{responsibilities}}

Qualifications:
{{qualifications}}

Resume:
{{resume}}""".format(corpus_warning=_CORPUS_WARNING)


def explain(resume: str, job: dict) -> tuple[str, bool]:
    logger.info("Generating explanation for %s at %s", job.get("title"), job.get("company"))

    try:
        responsibilities = json.loads(job.get("responsibilities", "[]"))
        qualifications = json.loads(job.get("qualifications", "[]"))
    except (json.JSONDecodeError, TypeError):
        responsibilities, qualifications = [], []

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )
    chain = ChatPromptTemplate.from_messages([("user", _PROMPT)]) | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "responsibilities": "\n".join(f"- {r}" for r in responsibilities),
            "qualifications": "\n".join(f"- {q}" for q in qualifications),
            "resume": resume,
        })
    except Exception:
        logger.warning("LLM call failed for %s at %s", job.get("title"), job.get("company"))
        return _CORPUS_WARNING, True

    result = result.strip()
    if not result or result == _CORPUS_WARNING:
        return _CORPUS_WARNING, True

    logger.info("Explanation generated for %s at %s", job.get("title"), job.get("company"))
    return result, False
