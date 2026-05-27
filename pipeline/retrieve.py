import logging

import chromadb

from config import CHROMA_COLLECTION, CHROMA_DIR, TOP_K_RETRIEVE

logger = logging.getLogger(__name__)


def retrieve(query_vector: list[float], top_k: int = TOP_K_RETRIEVE, filters: dict | None = None) -> list[dict]:
    logger.info("Retrieving top-%d candidates from ChromaDB", top_k)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(CHROMA_COLLECTION)

    kwargs = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
        "include": ["distances", "metadatas"],
    }
    if filters is not None:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    candidates = [
        {"job_id": job_id, "distance": distance, **metadata}
        for job_id, distance, metadata in zip(
            results["ids"][0], results["distances"][0], results["metadatas"][0]
        )
    ]
    logger.info("Retrieved %d candidates", len(candidates))
    return candidates
