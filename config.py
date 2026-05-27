import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
VOYAGE_API_KEY     = os.environ.get("VOYAGE_API_KEY")
COHERE_API_KEY     = os.environ.get("COHERE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# --- Paths ---
CHROMA_DIR        = os.environ.get("CHROMA_PATH", "chroma_index")
CHROMA_COLLECTION = "job_descriptions"

# --- Embedding ---
VOYAGE_MODEL = "voyage-3.5-lite"

# --- Retrieval ---
CHROMA_EF_SEARCH = 400  # HNSW ef_search; set at collection creation time in offline pipeline
TOP_K_RETRIEVE   = 100

# --- Reranking ---
COHERE_RERANK_MODEL  = "rerank-english-v3.0"
TOP_K_RERANK_DEFAULT = 10  # fallback if caller omits top_k

# --- Generation ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "deepseek/deepseek-chat-v3-0324"
