# Codebase Patterns & Design Reference

Patterns observed across the offline pipeline to follow when implementing the online pipeline.

---

## Module Structure

Each pipeline step is a single file with one public function. No classes unless state is truly needed.

```
pipeline/
  embed.py    → embed(text) -> list[float]
  retrieve.py → retrieve(vector, filters) -> list[dict]
  rerank.py   → rerank(query, candidates) -> list[dict]
  generate.py → generate(...) -> str
```

---

## Function Design

- One public entry point per module, named after what it produces
- Private helpers prefixed with `_`
- Explicit types on all signatures
- No side effects beyond the return value

**Offline example (`storage/embed.py`):**
```python
def embed_texts(texts: list[str], api_key: str) -> list[bytes]:
    embedder = VoyageAIEmbeddings(model=MODEL, api_key=api_key, batch_size=BATCH_SIZE)
    vectors = embedder.embed_documents(texts)
    return [np.array(vec, dtype=np.float32).tobytes() for vec in vectors]
```

---

## Retry / Backoff

Two patterns used in the offline pipeline:

**LangChain chains** — attach `.with_retry()` to the runnable:
```python
_chain = (prompt | llm.with_structured_output(Schema)).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

**SDK clients** — pass `max_retries` at construction:
```python
client = voyageai.Client(api_key=key, max_retries=3)  # default is 0
```

For the online pipeline, use `.with_retry()` on LangChain runnables (LLM steps) and `max_retries` on SDK clients (VoyageAI, Cohere).

---

## Config

All secrets and tunables live in `config.py`, loaded once via `python-dotenv`. Modules import directly from config — no passing config objects around.

```python
# config.py
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL   = "voyage-3.5-lite"
```

```python
# pipeline/embed.py
from config import VOYAGE_API_KEY, VOYAGE_MODEL
```

---

## Logging

```python
logger = logging.getLogger(__name__)
```

| Level | When |
|---|---|
| `INFO` | Top-level operation start/finish |
| `DEBUG` | Request-level internals, hard-to-trace paths |
| `WARNING` | Recoverable failures (retry, fallback triggered) |
| `ERROR` | Unrecoverable failures |

Don't log in straight-line code with obvious outcomes.

---

## Testing

- Mock external clients at the class level: `mocker.patch("module.ClassName", return_value=mock)`
- Verify contract properties, not implementation details:
  - Correct `input_type` passed to VoyageAI
  - Correct model name used
  - Retry config set
- Use `pytest-mock` (`mocker` fixture); already in `requirements.txt`

**Offline example (`tests/test_embed.py`):**
```python
def test_embed_texts_float32_roundtrip(mocker):
    mock_embedder = mocker.MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mocker.patch("storage.embed.VoyageAIEmbeddings", return_value=mock_embedder)
    result = embed_texts(["some text"], api_key="test-key")
    decoded = np.frombuffer(result[0], dtype=np.float32)
    assert np.allclose(decoded, [0.1, 0.2, 0.3], atol=1e-6)
```

---

## VoyageAI: Offline vs. Online

| | Offline | Online |
|---|---|---|
| Input type | `"document"` | `"query"` |
| Batching | 128 texts | Single string |
| Return type | `list[bytes]` (SQLite BLOB) | `list[float]` (ChromaDB query) |
| Retry | None (batch job, failures are fatal) | `max_retries=3` on `Client` |
| Library | `langchain_voyageai` | `voyageai` SDK directly |
