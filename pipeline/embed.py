import logging

import voyageai

from config import VOYAGE_API_KEY, VOYAGE_MODEL

logger = logging.getLogger(__name__)


def embed(text: str) -> list[float]:
    logger.info("Embedding resume text (%d chars)", len(text))
    client = voyageai.Client(api_key=VOYAGE_API_KEY, max_retries=3)
    result = client.embed([text], model=VOYAGE_MODEL, input_type="query")
    logger.info("Embedding complete, dim=%d", len(result.embeddings[0]))
    return result.embeddings[0]
